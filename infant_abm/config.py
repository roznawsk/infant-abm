from dataclasses import dataclass, asdict


@dataclass
class Config:
    persistence_boost_value: float = 0.0
    coordination_boost_value: float = 0.0

    def to_dict(self):
        return asdict(self)
