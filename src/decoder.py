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
        
        # 🚨 Tracks how many parameters we have generated!
        self.keys_generated: int = 0  

        self._colon_tokens = []
        for tid, tstr in self.manager.vocab.items():
            if tstr and all(c in ": \t\nĠ" for c in tstr):
                self._colon_tokens.append(tid)

    def get_valid_ids_for_current_state(self) -> List[int]:
        if self.state == "EXPECTING_KEY":
            return self.manager.get_string_token_ids()
        elif self.state == "EXPECTING_COLON":
            return self._colon_tokens
        elif self.state == "EXPECTING_VALUE":
            valid_ids = self.manager.get_number_token_ids()
            
            # 🚨 THE KILL SHOT: If we have generated all expected keys, ban the comma!
            if self.keys_generated >= len(self.expected_keys):
                valid_ids = [tid for tid in valid_ids if "," not in self.manager.vocab.get(tid, "")]
                
            return valid_ids
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
            # Check if a quote closed the key in this exact token
            if '"' in token_str and self.generated_text.count('"') % 2 == 0:
                # Extract the key safely
                parts = self.generated_text.split('"')
                if len(parts) >= 3:
                    self.current_key = parts[-2]
                
                # 🚨 THE FIX: Did this token already include the colon?
                # Check the text strictly AFTER the final closing quote
                after_quote = self.generated_text[self.generated_text.rfind('"')+1:]
                
                if ":" in after_quote:
                    self.state = "EXPECTING_VALUE"
                    self.keys_generated += 1
                else:
                    self.state = "EXPECTING_COLON"
                    
        elif self.state == "EXPECTING_COLON":
            if ":" in token_str:
                self.state = "EXPECTING_VALUE"
                self.keys_generated += 1
                
        elif self.state == "EXPECTING_VALUE":
            if "," in token_str:
                self.state = "EXPECTING_KEY" 
            elif "}" in token_str:
                self.is_finished = True