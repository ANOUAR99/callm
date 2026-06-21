import argparse
import json
import os
import numpy as np

from typing import List, Dict, Any


from src.llm_manager import LLMManager
from src.parser import load_function_definition, load_input_prompts
from src.decoder import ConstraintDecoder


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM Function Calling Engine")
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
    )
    parser.add_argument(
        "--input", type=str, default="data/input/function_calling_tests.json"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
    )
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

        available_funcs_str = ", ".join([f.name for f in functions])
        llm_prompt = (
            f"Task: Route the user's request to the correct function.\n"
            f"Available Functions: {available_funcs_str}\n\n"
            f"Request: 'Add 5 and 10'\nFunction: fn_add_numbers\n"
            f"Request: '{prompt_obj.prompt}'\nFunction: "
        )

        input_ids_tensor = manager.model.encode(llm_prompt)
        input_ids: List[int] = []
        if hasattr(input_ids_tensor, "tolist"):
            input_ids_raw = input_ids_tensor.tolist()
            input_ids = (
                input_ids_raw[0]
                if (
                    len(input_ids_raw) > 0
                    and isinstance(input_ids_raw[0], list)
                )
                else input_ids_raw
            )
        else:
            input_ids = list(input_ids_tensor)

        chosen_function_name = ""

        while True:
            logits = np.array(
                manager.model.get_logits_from_input_ids(input_ids)
            ).flatten()
            best_token_id = int(np.argmax(logits))
            best_str = manager.model.decode([best_token_id])

            if (
                best_str.strip() == ""
                or "{" in best_str
                or "\n" in best_str
                or "(" in best_str
            ):
                break

            chosen_function_name += best_str
            input_ids.append(best_token_id)

        chosen_function_name = (
            chosen_function_name.replace("Ġ", "")
            .replace(" ", "")
            .replace("'", "")
            .replace('"', "")
            .strip()
        )
        print(f"-> LLM selected function: '{chosen_function_name}'")

        try:
            chosen_func_def = next(
                f for f in functions if f.name == chosen_function_name
            )
        except StopIteration:
            print(
                f"⚠️ Error: LLM hallucinated an unknown function "
                f"'{chosen_function_name}'. Skipping prompt."
            )
            continue

        expected_keys = list(chosen_func_def.parameters.keys())
        schema_types = {
            k: v["type"] if isinstance(v, dict) else v.type
            for k, v in chosen_func_def.parameters.items()
        }
        decoder = ConstraintDecoder(manager, expected_keys, schema_types)

        context = "You are a function calling system.\nGiven a user request, identify the correct function to call.\nAvailable functions:"
        for f in functions:
            context += f"\n  - {f.name}: {f.description}\n"
            for pname, pschema in f.parameters.items():
                context += f"      {pname}: {pschema['type']}\n"
        context += "\n"

        json_prompt = context + "\n"
        json_prompt += "Generate only the parameter values.\nDo not generate JSON keys.\n"
        # Prevent the LLM from executing the function itself
        json_prompt += "Do not execute the function yourself. Only extract the raw arguments from the request.\n"

        json_prompt += f"Function: {chosen_function_name}\n"
        json_prompt += f"Expected Keys: {expected_keys}\n"
        json_prompt += f"Request: ```{prompt_obj.prompt}```\n"
        json_prompt += "JSON:\n"

        input_ids_tensor = manager.model.encode(json_prompt)
        input_ids = []
        if hasattr(input_ids_tensor, "tolist"):
            input_ids_raw = input_ids_tensor.tolist()
            input_ids = (
                input_ids_raw[0]
                if (
                    len(input_ids_raw) > 0
                    and isinstance(input_ids_raw[0], list)
                )
                else input_ids_raw
            )
        else:
            input_ids = list(input_ids_tensor)

        print("-> Generated Parameters: ", end="", flush=True)
        max_tokens = 50
        tokens_generated = 0

        generated_json = "{"

        for idx, key in enumerate(expected_keys):
            decoder.current_key_index = idx
            expected_type = schema_types[key]

            # 1. Explicitly handle leading quotes for strings
            if expected_type == "string":
                generated_json += f'"{key}": "'  # Inject the leading quote manually
            else:
                generated_json += f'"{key}": '

            current_context = json_prompt + generated_json

            input_ids_tensor = manager.model.encode(current_context)
            if hasattr(input_ids_tensor, "tolist"):
                input_ids_raw = input_ids_tensor.tolist()
                input_ids = (
                    input_ids_raw[0]
                    if (len(input_ids_raw) > 0 and isinstance(input_ids_raw[0], list))
                    else input_ids_raw
                )
            else:
                input_ids = list(input_ids_tensor)

            value_text = ""
            tokens_generated = 0

            while tokens_generated < 25: # Slightly increased for safety
                logits = np.array(manager.model.get_logits_from_input_ids(input_ids)).flatten()
                valid_ids = decoder.get_valid_ids_for_current_state()
                logits = decoder.apply_filter(logits, valid_ids)
                
                best_token_id = int(np.argmax(logits))
                token_str = manager.model.decode([best_token_id])
                
                value_text += token_str
                input_ids.append(best_token_id)
                tokens_generated += 1

                # 2. Updated breaking conditions
                if expected_type == "number":
                    if any(x in token_str for x in [",", "}", "\n"]):
                        break
                else: # string
                    # We only need ONE quote to close the string now
                    if '"' in token_str: 
                        break

            value_text = value_text.strip()

            # 3. Clean up the parsed values based on type
            if expected_type == "number":
                for stop in [",", "}", "\n"]:
                    if stop in value_text:
                        value_text = value_text.split(stop)[0]
                generated_json += value_text
            else:
                # For strings, grab everything up to the first closing quote
                if '"' in value_text:
                    value_text = value_text.split('"')[0]
                generated_json += value_text + '"' # Close the string

            # Add comma or close the JSON block
            if idx < len(expected_keys) - 1:
                generated_json += ", "
            else:
                generated_json += "}"

        print(generated_json)
        try:
            clean_json_str = generated_json
            if "}" in clean_json_str:
                clean_json_str = clean_json_str[: clean_json_str.find("}") + 1]

            parsed_params = json.loads(clean_json_str)

            for key, val in parsed_params.items():
                if schema_types.get(key) == "number":
                    try:
                        parsed_params[key] = float(val)
                    except (ValueError, TypeError):
                        pass

                final_results.append({
                    "prompt": prompt_obj.prompt,
                    "name": chosen_function_name,
                    "parameters": parsed_params,
                })
        except json.JSONDecodeError:
            print(
                f"⚠️ Error: Decoder generated invalid JSON. String was: "
                f"{clean_json_str}"
            )

    try:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=4)
        print(
            f"\nSuccessfully processed {len(prompts)} prompts. Results saved "
            f"to {args.output}"
        )
    except Exception as e:
        print(f"Failed to write output: {e}")


if __name__ == "__main__":
    main()
