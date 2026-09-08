"""Unit tests for serverRouter.smartRouter.param_types.

Pure-function tests: no FastAPI app, no network, no Firestore, no credentials.
"""
from serverRouter.smartRouter.param_types import CostType, LatencyType


def _raises_value_error(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return True
    return False


def test_named_preference_returns_enum_member():
    assert LatencyType.from_value("balanced") is LatencyType.BALANCED
    assert LatencyType.from_value("LIGHTNING") is LatencyType.LIGHTNING
    assert CostType.from_value("cheap") is CostType.CHEAP


def test_enum_member_passes_through():
    assert LatencyType.from_value(LatencyType.FAST) is LatencyType.FAST
    assert CostType.from_value(CostType.PREMIUM) is CostType.PREMIUM


def test_numeric_value_is_accepted():
    assert LatencyType.from_value(2.5) == 2.5
    assert CostType.from_value(12) == 12.0
    assert isinstance(LatencyType.from_value(12), float)


def test_numeric_string_is_accepted():
    # Regression: numeric strings previously raised ValueError -> HTTP 500 on
    # /v1/smartRouter even though a numeric limit is a documented input.
    assert LatencyType.from_value("2.5") == 2.5
    assert CostType.from_value("0.75") == 0.75


def test_invalid_string_raises_value_error():
    for bad in ("superfast", "", "cheapish", "none"):
        assert _raises_value_error(LatencyType.from_value, bad), bad
        assert _raises_value_error(CostType.from_value, bad), bad


def test_bool_is_rejected():
    assert _raises_value_error(LatencyType.from_value, True)
    assert _raises_value_error(CostType.from_value, False)


def test_conversion_yields_numeric_constraint_for_ranking():
    # Mirrors the constraint extraction in serverRouter/smartRouter/main.py;
    # the else-branch must use the *converted* value, not the raw input.
    lat = LatencyType.from_value("2.5")
    cost = CostType.from_value("balanced")
    lat_value = lat.value_in_seconds if isinstance(lat, LatencyType) else lat
    cost_value = cost.value_in_dollars if isinstance(cost, CostType) else cost
    assert isinstance(lat_value, float) and lat_value == 2.5
    assert isinstance(cost_value, float) and cost_value == 10.0
