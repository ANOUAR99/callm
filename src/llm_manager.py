import json
import sys
from typing import Dict
from llm_sdk import Small_LLM_Model


class LLMManager:
    def __init__(self):
        print("Initializing LLM Model...")
        self.model = Small_LLM_Model()
        self.vocab = self._load_vocabulary()
        print(len(self.vocab))

    def _load_vocabulary(self) -> Dict[int, str]:
        vocab_path = self.model.get_path_to_vocab_file()
        parsed_vocab = {}

        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                raw_vocab = json.load(f)
                for key, value in raw_vocab.items():
                    parsed_vocab[int(value)] = key
        except FileNotFoundError as e:
            print(e)
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(e)
            sys.exit(1)
        return parsed_vocab

    def get_number_token_ids(self) -> list[int]:
        valid_ids = []
        
        allowed_chars = set("0123456789.- Ġ")

        for token_id, token_str in self.vocab.items():
            if token_str and all(char in allowed_chars for char in token_str):
                valid_ids.append(token_id)
                
        return valid_ids
    