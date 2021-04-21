from abc import ABC
from typing import Dict, Any, Callable


class BaseContainer(ABC):
    """
        Base container class, for easier definition of containers. Ensures consistent format
        before being turned into a dataframe row.
    """

    def __init__(self, id: str, name: str, class_def: Callable, args: Dict[str, Any] = None):
        """
            class_def: The callable to be invoked
        """
        self.id = id
        self.name = name
        self.class_def = class_def
        self.args = args

    def get_dict(self, internal=False) -> Dict[str, Any]:
        d = [("ID", self.id), ("Name", self.name)]

        if internal:
            d += [
                ("Class", self.class_def),
                ("Args", self.args),
            ]

        return dict(d)
