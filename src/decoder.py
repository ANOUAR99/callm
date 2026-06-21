import numpy as np
from typing import List, Dict


class ConstraintDecoder:
    def __init__(
        self,
        manager,
        expected_keys: List[str],
        schema_types: Dict[str, str],
    ) -> None:
        self.manager = manager
        self.expected_keys = expected_keys
        self.schema_types = schema_types

        self.generated_text = ""
        self.current_key_index = 0
        self.is_finished = False

    @property
    def current_key(self) -> str:
        return self.expected_keys[self.current_key_index]

    def get_valid_ids_for_current_state(self) -> List[int]:
        expected_type = self.schema_types.get(
            self.current_key,
            "string",
        )

        if expected_type == "number":
            return self.manager.get_number_token_ids()

        return self.manager.get_string_token_ids()

    def apply_filter(
        self,
        logits: np.ndarray,
        valid_ids: List[int],
    ) -> np.ndarray:
        if not valid_ids:
            return logits

        mask = np.ones(logits.size, dtype=bool)
        mask[valid_ids] = False
        logits[mask] = -np.inf
        return logits