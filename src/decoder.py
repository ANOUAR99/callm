import numpy as np
from typing import List, Dict
from src.llm_manager import LLMManager

class ConstraintDecoder:
    """A State Machine enforcing strict JSON generation rules."""
    def __init__(self, manager: LLMManager, expected_keys: List[str], schema_types: Dict[str, str]) -> None:
        self.manager = manager
        self.expected_keys = expected_keys
        self.schema_types = schema_types
        
        self.generated_text: str = ""
        self.is_finished: bool = False
        self.state: str = "EXPECTING_KEY"

    def get_valid_ids_for_current_state(self) -> List[int]:
        if self.state in ["EXPECTING_KEY", "EXPECTING_COLON"]:
            return self.manager.get_string_token_ids()
        elif self.state == "EXPECTING_VALUE":
            return self.manager.get_number_token_ids()
        return []

    def apply_filter(self, logits: np.ndarray, valid_ids: List[int]) -> np.ndarray:
        if not valid_ids:
            return logits
            
        mask = np.ones(logits.size, dtype=bool)
        mask[valid_ids] = False
        logits[mask] = -np.inf
        return logits

    def update_state(self, token_str: str) -> None:
        self.generated_text += token_str
        
        if self.state == "EXPECTING_KEY":
            # We count the quotes! If there is an EVEN number of quotes, 
            # it means the key is closed and we are ready for a colon.
            if '"' in token_str and self.generated_text.count('"') % 2 == 0:
                self.state = "EXPECTING_COLON"
                
        if self.state == "EXPECTING_COLON":
            # Removed the generated_text check so it works for multiple parameters!
            if ":" in token_str:
                self.state = "EXPECTING_VALUE"
                
        if self.state == "EXPECTING_VALUE":
            if "," in token_str:
                self.state = "EXPECTING_KEY" 
            if "}" in token_str:
                self.is_finished = True