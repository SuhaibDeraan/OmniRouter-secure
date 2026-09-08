from enum import Enum
from typing import Union, Dict, Any

class LatencyType(str, Enum):
    """
    Defines latency preference types that users can select.
    Each type corresponds to a maximum latency value for the model.

    Latency is measured in seconds to first token.
    """
    LIGHTNING = "lightning"
    FAST = "fast"
    BALANCED = "balanced"
    PERFORMANCE = "performance"

    @property
    def value_in_seconds(self) -> float:
        mapping = {
            LatencyType.LIGHTNING: 0.5,
            LatencyType.FAST: 0.8,
            LatencyType.BALANCED: 1.5,
            LatencyType.PERFORMANCE: 5.0
        }
        return mapping[self]

    @classmethod
    def from_value(cls, value: Union[str, float, "LatencyType"]) -> Union[float, "LatencyType"]:
        """Convert an enum member, preference name, or number to a usable value.

        Accepts an enum member, one of the preference names (case-insensitive),
        or a raw number / numeric string (used as an explicit limit). Any other
        value raises ValueError so the caller can return a 4xx.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError(f"Invalid latency type: {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                valid_options = [e.name for e in cls]
                raise ValueError(
                    f"Invalid latency type: {value}. "
                    f"Choose from {valid_options} or pass a number"
                )
        raise ValueError(f"Invalid latency type: {value!r}")


class CostType(str, Enum):
    """
    Defines cost preference types that users can select.
    Each type corresponds to a maximum cost value for the model.

    Cost is measured in dollars per million tokens.
    """
    CHEAP = "cheap"
    BALANCED = "balanced"
    PREMIUM = "premium"
    PERFORMANCE = "performance"

    @property
    def value_in_dollars(self) -> float:
        mapping = {
            CostType.CHEAP: 1.5,
            CostType.BALANCED: 10.0,
            CostType.PREMIUM: 30.0,
            CostType.PERFORMANCE: 100.0
        }
        return mapping[self]

    @classmethod
    def from_value(cls, value: Union[str, float, "CostType"]) -> Union[float, "CostType"]:
        """Convert an enum member, preference name, or number to a usable value.

        Accepts an enum member, one of the preference names (case-insensitive),
        or a raw number / numeric string (used as an explicit limit). Any other
        value raises ValueError so the caller can return a 4xx.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, bool):
            raise ValueError(f"Invalid cost type: {value!r}")
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return cls(value.lower())
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                valid_options = [e.name for e in cls]
                raise ValueError(
                    f"Invalid cost type: {value}. "
                    f"Choose from {valid_options} or pass a number"
                )
        raise ValueError(f"Invalid cost type: {value!r}")


