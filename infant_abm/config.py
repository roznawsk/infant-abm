from dataclasses import dataclass


@dataclass
class Config:
    persistence_boost_value: float
    coordination_boost_value: float
