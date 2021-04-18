from abc import ABC
from typing import Dict, Any


class BaseContainer(ABC):
    """
        Base container class, for easier definition of containers. Ensures consistent format
        before being turned into a dataframe row.
    """

    def __init__(self, id: str, name: str, class_def: str, args: Dict[str, Any] = None):
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
