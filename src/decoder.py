import numpy as np
from src.llm_manager import LLMManager

class ConstraintDecoder:
    def __init__(self, manager: LLMManager, expected_keys: list[str], schema_types: dict):
        self.manager = manager
        self.expected_keys = expected_keys
        self.schema_types = schema_types 
        
        # State Tracking Variables
        self.current_key_index = 0
        self.generated_text = ""
        self.is_finished = False

    def get_valid_ids_for_current_state(self) -> list[int]:
        """Looks at the current key and asks the manager for the allowed IDs."""
        
        # If we have generated all keys, the only valid token left is the closing brace
        if self.current_key_index >= len(self.expected_keys):
            # We look for "}" in the vocab
            return [id for id, text in self.manager.vocab.items() if "}" in text]
            
        current_key = self.expected_keys[self.current_key_index]
        expected_type = self.schema_types[current_key]
        
        # Ask the manager for the right IDs based on the expected type
        if expected_type == "number":
            return self.manager.get_number_token_ids()
        # You can add boolean or string checks here later if needed!
        # elif expected_type == "boolean":
        #     return self.manager.get_boolean_token_ids()
        
        # Fallback: if we don't recognize the type, just allow everything (empty list means no filter)
        return []

    def apply_filter(self, logits: np.ndarray, valid_ids: list[int]) -> np.ndarray:
        """Crushes all invalid logits to -inf using high-speed NumPy masking."""
        if not valid_ids:
            return logits # If no specific valid IDs, return logits unmodified
            
        mask = np.ones(logits.size, dtype=bool)
        mask[valid_ids] = False
        logits[mask] = -np.inf
        
        return logits

    def update_state(self, new_token_str: str):
        """Updates the running text and checks if we need to move to the next key."""
        self.generated_text += new_token_str
        
        # If the token contains a comma, the current field is done. Advance the machine!
        if "," in new_token_str:
            self.current_key_index += 1
            
        # If the token contains a closing brace, the entire JSON is done!
        if "}" in new_token_str:
            self.is_finished = True