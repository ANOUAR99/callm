import argparse
import json
import sys
import numpy as np
import torch
from typing import List

print(f"🔥 GPU WAKEUP CHECK: {torch.cuda.is_available()}")

from src.llm_manager import LLMManager
from src.parser import load_function_definition, load_input_prompts
from src.decoder import ConstraintDecoder
from src.models import FunctionDefinition

def parse_arguments():
    """Handles the command-line arguments required by the project."""
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    parser.add_argument("--functions_definition", type=str, default="data/input/function_definitions.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("Loading definitions and prompts...")
    functions = load_function_definition(args.functions_definition)
    prompts = load_input_prompts(args.input)
    
    print("Initializing LLM Model...")
    manager = LLMManager()
    
    final_results = []
    
    for prompt_obj in prompts:
        print(f"\nProcessing prompt: {prompt_obj.prompt}")
        llm_prompt = f"Given the prompt '{prompt_obj.prompt}', which function should be called? Return only the function name."
        
        # ==========================================
        # PHASE A: Pick the Function
        # ==========================================
        input_ids = manager.model.encode(llm_prompt)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            
        chosen_function_name = ""
        
        while True:
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            best_token_id = int(np.argmax(logits)) 
            best_str = manager.model.decode([best_token_id])
            
            if best_str.strip() == "" or "{" in best_str or "\n" in best_str or "(" in best_str:
                break
                
            chosen_function_name += best_str
            input_ids.append(best_token_id)
            
        # The Bulldozer (Safely un-indented!)
        chosen_function_name = chosen_function_name.replace("Ġ", "").replace(" ", "").strip()
        print(f"-> LLM selected function: '{chosen_function_name}'")
        
        # ==========================================
        # PHASE B: Set up the Decoder
        # ==========================================
        try:
            chosen_func_def = next(f for f in functions if f.name == chosen_function_name)
        except StopIteration:
            print(f"⚠️ Error: LLM hallucinated an unknown function '{chosen_function_name}'. Skipping prompt.")
            continue 
        
        expected_keys = list(chosen_func_def.parameters.keys()) 
        schema_types = {k: v["type"] for k, v in chosen_func_def.parameters.items()} 
        
        decoder = ConstraintDecoder(manager, expected_keys, schema_types)
        
        # ==========================================
        # PHASE C: The Constrained JSON Loop
        # ==========================================
        json_prompt = f"Generate JSON parameters for the function {chosen_function_name} based on the prompt: '{prompt_obj.prompt}'. Parameters: {expected_keys}. Output valid JSON starting with {{"
        
        input_ids = manager.model.encode(json_prompt)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
            
        print("-> Generated Parameters: {", end="", flush=True)
        
        max_tokens = 50 # THE KILL SWITCH
        tokens_generated = 0

        while not decoder.is_finished and tokens_generated < max_tokens:
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            
            valid_ids = decoder.get_valid_ids_for_current_state()
            logits = decoder.apply_filter(logits, valid_ids)
            
            best_token_id = int(np.argmax(logits))
            best_token_str = manager.model.decode([best_token_id])
            
            # The real-time stream print
            print(best_token_str, end="", flush=True)
            
            decoder.update_state(best_token_str)
            input_ids.append(best_token_id)
            tokens_generated += 1
            
        print() # Clean new line when JSON completes
        
        # ==========================================
        # PHASE D: Save the Results
        # ==========================================
        try:
            # We add the opening bracket manually since we streamed it to the terminal!
            clean_json_str = "{" + decoder.generated_text
            parsed_params = json.loads(clean_json_str)
            
            final_results.append({
                "prompt": prompt_obj.prompt,
                "name": chosen_function_name,
                "parameters": parsed_params
            })
        except json.JSONDecodeError:
            print(f"⚠️ Error: Decoder generated invalid JSON for {chosen_function_name}. Skipping save.")

    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nSuccessfully processed {len(prompts)} prompts. Results saved to {args.output}")

if __name__ == "__main__":
    main()