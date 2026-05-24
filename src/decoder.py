import numpy as np

class ConstraintDecoder:
    def __init__(self, manager, expected_keys, schema_types):
        self.manager = manager
        self.expected_keys = expected_keys
        self.schema_types = schema_types
        
        self.generated_text = ""
        self.is_finished = False
        
        # The internal State Machine trackers
        self.state = "EXPECTING_KEY"

    def get_valid_ids_for_current_state(self) -> list[int]:
        """Asks the manager for the allowed tokens based on the current state."""
        # Note: This assumes you added get_string_token_ids to your manager! 
        # If not, you can just return manager.get_number_token_ids() as a fallback.
        if self.state == "EXPECTING_KEY" or self.state == "EXPECTING_COLON":
            if hasattr(self.manager, "get_string_token_ids"):
                return self.manager.get_string_token_ids()
            return [] # Fallback: let the AI guess if no string list exists
            
        elif self.state == "EXPECTING_VALUE":
            return self.manager.get_number_token_ids()
            
        return []

    def apply_filter(self, logits: np.ndarray, valid_ids: list[int]) -> np.ndarray:
        """Applies an infinite negative mask to any token not in valid_ids."""
        if not valid_ids:
            return logits # Safety net: if list is empty, don't filter
            
        mask = np.ones(logits.size, dtype=bool)
        mask[valid_ids] = False
        logits[mask] = -np.inf
        return logits

    def update_state(self, token_str: str):
        """Moves the state machine forward based on the punctuation the AI generated."""
        self.generated_text += token_str
        
        if self.state == "EXPECTING_KEY":
            # If we see a closing quote (and we've generated more than just the opening bracket)
            if '"' in token_str and len(self.generated_text.strip()) > 2:
                self.state = "EXPECTING_COLON"
                
        elif self.state == "EXPECTING_COLON":
            if ":" in token_str:
                self.state = "EXPECTING_VALUE"
                
        elif self.state == "EXPECTING_VALUE":
            if "," in token_str:
                self.state = "EXPECTING_KEY" # Loop back for the next parameter!
            elif "}" in token_str:
                self.is_finished = True      # We are completely done!