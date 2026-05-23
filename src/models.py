
from pydantic import BaseModel

class Funcdef(BaseModel):
    name: str
    description: str
    parameters: dict[str, dict[str, str]]
    returns: dict[str, str]


class PromptInput(BaseModel):
    prompt: str

from typing import Any

class Funcresults(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]



