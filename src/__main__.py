import argparse
import json
import sys
import numpy as np

from src.llm_manager import LLMManager
from src.parser import load_function_definition, load_input_prompts
from src.decoder import ConstraintDecoder

def parse_arguments():
    """Handles the command-line arguments required by the project."""
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    
    # Setting up the exact flags requested in the assignment with their default paths
    parser.add_argument("--functions_definition", type=str, 
                        default="data/input/function_definitions.json")
    parser.add_argument("--input", type=str, 
                        default="data/input/function_calling_tests.json")
    parser.add_argument("--output", type=str, 
                        default="data/output/function_calling_results.json")
    
    return parser.parse_args()

def main():
    # 1. Parse command line arguments
    args = parse_arguments()

    # 2. Ingest the Data (Using your parser.py)
    print("Loading definitions and prompts...")
    functions = load_function_definition(args.functions_definition)
    prompts = load_input_prompts(args.input)

    # 3. Initialize the Engine (Using your manager.py)
    manager = LLMManager()
    
    final_results = []

    # 4. Process each prompt one by one
    for prompt_obj in prompts:
        print(f"\nProcessing prompt: {prompt_obj.prompt}")
        
        # We need to build the specific prompt to feed the LLM here.
        # It needs to include the available functions so the LLM knows what its options are!
        llm_prompt = f"Available functions: {[f.name for f in functions]}\nUser: {prompt_obj.prompt}\nCall:"
        
        # ==========================================
        # PHASE A: Let the LLM pick the function name
        # ==========================================
        input_ids = manager.model.encode(llm_prompt)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist() # Convert tensor to list
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0] # Flatten it if it's 2D
        chosen_function_name = ""
        
        while True:
            # Force the SDK's list into a flat NumPy array!
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            best_token_id = int(np.argmax(logits)) 
            best_str = manager.model.decode([best_token_id])
            
            # We added "(" to the stop conditions to prevent Python-style function calls!
            if best_str.strip() == "" or "{" in best_str or "\n" in best_str or "(" in best_str:
                break
                
            chosen_function_name += best_str
            input_ids.append(best_token_id)
        chosen_function_name = chosen_function_name.replace("Ġ", "").replace(" ", "").strip()
        print(f"-> LLM selected function: {chosen_function_name}")
        
        # ==========================================
        # PHASE B: Set up the Decoder
        # ==========================================
        try:
            # Find the matching blueprint from our parsed data
            chosen_func_def = next(f for f in functions if f.name == chosen_function_name)
        except StopIteration:
            print(f"⚠️ Error: LLM hallucinated an unknown function '{chosen_function_name}'. Skipping prompt.")
            continue # This instantly stops the current loop and moves to the next prompt!
        
        expected_keys = list(chosen_func_def.parameters.keys()) 
        schema_types = {k: v["type"] for k, v in chosen_func_def.parameters.items()} 
        
        decoder = ConstraintDecoder(manager, expected_keys, schema_types)
        
        # ==========================================
        # PHASE C: The Constrained JSON Loop
        # ==========================================
        # Force the model to start writing the parameters object by adding "{"
        json_prompt = llm_prompt + chosen_function_name + "\n{"
        input_ids = manager.model.encode(json_prompt)
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        if len(input_ids) > 0 and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        
        while not decoder.is_finished:
            # Force the SDK's list into a flat NumPy array!
            logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
            
            # 1. Ask decoder for rules
            valid_ids = decoder.get_valid_ids_for_current_state()
            
            # 2. Filter the logits using our high-speed NumPy mask
            logits = decoder.apply_filter(logits, valid_ids)
            
            # 3. Pick the winner
            best_token_id = int(np.argmax(logits))
            best_token_str = manager.model.decode([best_token_id])
            
            # 4. Update our state tracker
            decoder.update_state(best_token_str)
            
            # 5. Add to history for the next step
            input_ids.append(best_token_id)
            
        print(f"-> Generated Parameters: {{{decoder.generated_text}")
        
        # ==========================================
        # PHASE D: Save the Results
        # ==========================================
        # Because of your Constraint Decoder, you can safely parse it as JSON!
        final_results.append({
            "prompt": prompt_obj.prompt,
            "name": chosen_function_name,
            "parameters": json.loads("{" + decoder.generated_text)
        })

    # 5. Write the final 100% valid JSON to the output file
    with open(args.output, "w") as f:
        json.dump(final_results, f, indent=4)
        
    print(f"\nSuccessfully processed {len(prompts)} prompts. Results saved to {args.output}")

if __name__ == "__main__":
    main()