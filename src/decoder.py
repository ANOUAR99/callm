import numpy as np
from typing import List, Dict

class ConstraintDecoder:
    """A State Machine enforcing strict JSON generation rules."""
    def __init__(self, manager, expected_keys: List[str], schema_types: Dict[str, str]) -> None:
        self.manager = manager
        self.expected_keys = expected_keys
        self.schema_types = schema_types
        
        self.generated_text: str = ""
        self.is_finished: bool = False
        self.state: str = "EXPECTING_KEY"

        # 🚨 NEW: Find every token that is strictly a colon or a space!
        self._colon_tokens = []
        for tid, tstr in self.manager.vocab.items():
            if tstr and all(c in ": \t\nĠ" for c in tstr):
                self._colon_tokens.append(tid)

    def get_valid_ids_for_current_state(self) -> List[int]:
        if self.state == "EXPECTING_KEY":
            return self.manager.get_string_token_ids()
        elif self.state == "EXPECTING_COLON":
            # 🚨 THE FIX: Force the colon! No letters or numbers allowed!
            return self._colon_tokens
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
            # Check for an EVEN number of quotes to ensure the key is completely closed
            if '"' in token_str and self.generated_text.count('"') % 2 == 0:
                self.state = "EXPECTING_COLON"
                
        # We use 'if' so the machine can jump multiple states on multi-character tokens (like '": ')
        if self.state == "EXPECTING_COLON":
            if ":" in token_str:
                self.state = "EXPECTING_VALUE"
                
        if self.state == "EXPECTING_VALUE":
            if "," in token_str:
                self.state = "EXPECTING_KEY" 
            elif "}" in token_str:
                self.is_finished = True