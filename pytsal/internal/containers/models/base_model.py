from abc import ABC
from typing import Dict, Any

from pytsal.internal.containers.base_container import BaseContainer


class ModelContainer(BaseContainer, ABC):
    def __init__(self, id: str, name: str, model: Any, args: Dict[str, Any] = None):
        super().__init__(id, name, model, args=args)
