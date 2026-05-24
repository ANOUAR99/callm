import argparse
import json
import os
import numpy as np
import torch
from typing import List, Dict, Any

print(f"🔥 GPU WAKEUP CHECK: {torch.cuda.is_available()}")

from src.llm_manager import LLMManager
from src.parser import load_function_definition, load_input_prompts
from src.decoder import ConstraintDecoder

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    parser.add_argument("--functions_definition", type=str, default="data/input/function_definitions.json")
    parser.add_argument("--input", type=str, default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, default="data/output/function_calling_results.json")
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()
    
    print("Loading definitions and prompts...")
    functions = load_function_definition(args.functions_definition)
    prompts = load_input_prompts(args.input)
    
    if not functions or not prompts:
        print("Data files missing or empty. Exiting.")
        return
        
    print("Initializing LLM Model...")
    manager = LLMManager()
    final_results: List[Dict[str, Any]] = []
    
    for prompt_obj in prompts:
        print(f"\nProcessing prompt: {prompt_obj.prompt}")
        
        # We use .join() to create a clean string without Python's list syntax
        available_funcs_str = ", ".join([f.name for f in functions])
        # Few-Shot Pattern: Show it exactly how to answer!
        llm_prompt = (
            f"Task: Route the user's request to the correct function.\n"
            f"Available Functions: {available_funcs_str}\n\n"
            f"Request: 'Add 5 and 10'\nFunction: fn_add_numbers\n"
            f"Request: '{prompt_obj.prompt}'\nFunction: "
        )
        
        # ==========================================
        # Phase A: Router
        # ==========================================
        input_ids_tensor = manager.model.encode(llm_prompt)
        input_ids: List[int] = []
        if hasattr(input_ids_tensor, "tolist"):
            input_ids_raw = input_ids_tensor.tolist()
            input_ids = input_ids_raw[0] if (len(input_ids_raw) > 0 and isinstance(input_ids_raw[0], list)) else input_ids_raw
        else:
            input_ids = list(input_ids_tensor)
            
        chosen_function_name = ""
        
        while True:
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            best_token_id = int(np.argmax(logits)) 
            best_str = manager.model.decode([best_token_id])
            
            if best_str.strip() == "" or "{" in best_str or "\n" in best_str or "(" in best_str:
                break
                
            chosen_function_name += best_str
            input_ids.append(best_token_id)
            
        # The Upgraded Bulldozer: Now crushes both single and double quotes!
        chosen_function_name = chosen_function_name.replace("Ġ", "").replace(" ", "").replace("'", "").replace('"', "").strip()
        print(f"-> LLM selected function: '{chosen_function_name}'")
        
        # ==========================================
        # Phase B: Initialization
        # ==========================================
        try:
            chosen_func_def = next(f for f in functions if f.name == chosen_function_name)
        except StopIteration:
            print(f"⚠️ Error: LLM hallucinated an unknown function '{chosen_function_name}'. Skipping prompt.")
            continue 
        
        expected_keys = list(chosen_func_def.parameters.keys()) 
        # Safely extract the type whether Pydantic made it an object or left it as a dict!
        schema_types = {k: v["type"] if isinstance(v, dict) else v.type for k, v in chosen_func_def.parameters.items()} 
        decoder = ConstraintDecoder(manager, expected_keys, schema_types)
        
        # ==========================================
        # Phase C: Generator
        # ==========================================
        # Few-Shot Pattern: Force it into a strict JSON mindset!
        json_prompt = (
            f"Task: Extract parameters as valid JSON.\n\n"
            f"Function: fn_add_numbers\n"
            f"Expected Keys: ['a', 'b']\n"
            f"Request: 'Add 5 and 10'\n"
            f"JSON:\n{{\"a\": 5, \"b\": 10}}\n\n"
            f"Function: {chosen_function_name}\n"
            f"Expected Keys: {expected_keys}\n"
            f"Request: '{prompt_obj.prompt}'\n"
            f"JSON:\n{{"
        )

        input_ids_tensor = manager.model.encode(json_prompt)
        input_ids = []
        if hasattr(input_ids_tensor, "tolist"):
            input_ids_raw = input_ids_tensor.tolist()
            input_ids = input_ids_raw[0] if (len(input_ids_raw) > 0 and isinstance(input_ids_raw[0], list)) else input_ids_raw
        else:
            input_ids = list(input_ids_tensor)
            
        print("-> Generated Parameters: {", end="", flush=True)
        max_tokens = 50 
        tokens_generated = 0

        while not decoder.is_finished and tokens_generated < max_tokens:
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            valid_ids = decoder.get_valid_ids_for_current_state()
            logits = decoder.apply_filter(logits, valid_ids)
            
            best_token_id = int(np.argmax(logits))
            best_token_str = manager.model.decode([best_token_id])
            print(best_token_str, end="", flush=True)
            
            decoder.update_state(best_token_str)
            input_ids.append(best_token_id)
            tokens_generated += 1
            
        print() 
        
        # ==========================================
        # Phase D: Extraction and Garbage Cleansing
        # ==========================================
        try:
            clean_json_str = "{" + decoder.generated_text
            if "}" in clean_json_str:
                clean_json_str = clean_json_str[:clean_json_str.find("}") + 1]
                
            parsed_params = json.loads(clean_json_str)
            
            final_results.append({
                "prompt": prompt_obj.prompt,
                "name": chosen_function_name,
                "parameters": parsed_params
            })
        except json.JSONDecodeError:
            print(f"⚠️ Error: Decoder generated invalid JSON. String was: {clean_json_str}")

    # ==========================================
    # IO Write
    # ==========================================
    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(f"\nSuccessfully processed {len(prompts)} prompts. Results saved to {args.output}")
    except Exception as e:
        print(f"Failed to write output: {e}")

if __name__ == "__main__":
    main()