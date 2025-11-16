from dataclasses import dataclass
from typing import Literal

Status = Literal["Active", "Pruned"]


@dataclass
class AttributeExplanation:
    name: str
    argumentation: str
    command: str
    added: str | int | None = None
    status: Status = "Active"

    @classmethod
    def from_dict(cls, data: dict) -> "AttributeExplanation":
        return cls(
            name=data["name"],
            argumentation=data["description"],
            command=data["pd_command"],
            added=data.get("added"),
            status=data.get("status", "Active"),
        )

    @property
    def as_dict(self) -> dict:
        return {
            "Attribute": self.name,
            "Description": self.argumentation,
            "Pandas Command": self.command,
            "Status": self.status,
            "Added": self.added,
        }

    def __repr__(self):
        return f"Description: {self.argumentation} - Pandas Command: {self.command}"
