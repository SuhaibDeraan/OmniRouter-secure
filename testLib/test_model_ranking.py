"""Unit tests for serverRouter.smartRouter.model_ranking.

These are pure-function tests: no FastAPI app, no network, no Firestore.
"""
from serverRouter.smartRouter.model_ranking import aggregate_model_metrics, rank_models

# A minimal task -> model metric table, shaped like database/_task_models.json
TASK_MODELS = {
    "coding": {
        "cheap-fast": {"accuracy": 0.6, "cost": 1.0, "latency": 0.5},
        "pricey-slow": {"accuracy": 0.9, "cost": 50.0, "latency": 10.0},
    }
}

EXPECTED_KEYS = {"model", "score", "cost", "latency", "message"}


def test_ranks_best_model_within_constraints():
    aggregated = aggregate_model_metrics({"coding": 1.0}, TASK_MODELS)
    result = rank_models(aggregated, max_latency=5.0, max_cost=10.0, model_list=[])

    assert set(result) == EXPECTED_KEYS
    assert result["model"] == "cheap-fast"  # pricey-slow is filtered out by cost/latency


def test_falls_back_to_default_when_no_model_meets_constraints():
    aggregated = aggregate_model_metrics({"coding": 1.0}, TASK_MODELS)
    # Constraints tighter than every candidate
    result = rank_models(aggregated, max_latency=0.1, max_cost=0.1, model_list=[])

    assert set(result) == EXPECTED_KEYS
    assert result["model"] == "gpt-4o-mini"
    assert "No models meet criteria" in result["message"]


def test_falls_back_to_default_when_classification_is_empty():
    # No similar tasks -> empty aggregation -> must still return the standard shape
    result = rank_models({}, max_latency=1.5, max_cost=10.0, model_list=[])

    assert set(result) == EXPECTED_KEYS
    assert result["model"] == "gpt-4o-mini"


def test_result_shape_is_consumable_by_main_router_formatting():
    # Mirrors the f-string access pattern in smartRouter/main.py
    result = rank_models({}, max_latency=1.5, max_cost=10.0, model_list=[])
    _ = f"{result['message']}"
    _ = (
        f"Selected {result['model']} with highest score, "
        f"{result['cost'] / 1000} $/k tokens, and {result['latency']} second latency"
    )
