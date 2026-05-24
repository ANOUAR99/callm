import json
import sys
from typing import List
from src.models import Funcdef, PromptInput

def load_function_definition(filepath: str) -> List[Funcdef]:
    parsed_function = []
    try:
        with open(filepath, "r") as f:
            raw_data = json.load(f)
            for item in raw_data:
                func = Funcdef(**item)
                parsed_function.append(func)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Error: The file '{filepath}' contains invalid JSON formatting.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Data validation failed in '{filepath}'.\nDetails: {e}")
        sys.exit(1)
    return parsed_function


def load_input_prompts(filepath: str) -> List[PromptInput]:
    parsed_prompts = []
    try:
        with open(filepath, "r") as f:
            raw_data = json.load(f)
            for item in raw_data:
                prompt_obj = PromptInput(**item)
                parsed_prompts.append(prompt_obj)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(
            f"Error: The file '{filepath}' contains invalid JSON formatting.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Data validation failed in '{filepath}'.\nDetails: {e}")
        sys.exit(1)
    return parsed_prompts