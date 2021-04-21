from abc import ABC
from typing import Dict, Any

from pytsal.internal.containers.base_container import BaseContainer


class MetricContainer(BaseContainer, ABC):
    def __init__(self, id: str, name: str, class_def: Any, args: Dict[str, Any] = None):
        super().__init__(id, name, class_def, args=args)
