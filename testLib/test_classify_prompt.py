"""Unit tests for serverRouter.smartRouter.classifyPrompt.classify_prompt.

The embedding-backed task_manager is stubbed, so this is a pure test of the
score-normalization logic. No openai, no pickle, no network.
"""
import importlib
import sys
import types

import pytest


class _FakeTaskManager:
    def __init__(self):
        self.result = []

    def find_similar_tasks(self, query, top_n=6):
        return list(self.result)


_fake_tm = _FakeTaskManager()
_TM_NAME = "serverRouter.smartRouter.taskEmbeddingManager"
_CP_NAME = "serverRouter.smartRouter.classifyPrompt"

# Force the real classifyPrompt module to load against our fake task_manager,
# regardless of whether another test file has stubbed classifyPrompt itself.
_saved_tm = sys.modules.get(_TM_NAME)
_saved_cp = sys.modules.get(_CP_NAME)
_stub = types.ModuleType(_TM_NAME)
_stub.task_manager = _fake_tm
sys.modules[_TM_NAME] = _stub
sys.modules.pop(_CP_NAME, None)

classify_prompt = importlib.import_module(_CP_NAME).classify_prompt  # noqa: E402


def _restore_modules():
    if _saved_cp is not None:
        sys.modules[_CP_NAME] = _saved_cp
    else:
        sys.modules.pop(_CP_NAME, None)
    if _saved_tm is not None:
        sys.modules[_TM_NAME] = _saved_tm
    else:
        sys.modules.pop(_TM_NAME, None)


@pytest.fixture(autouse=True)
def _reset():
    _fake_tm.result = []
    yield
    _fake_tm.result = []


def teardown_module(module):
    _restore_modules()


def test_empty_similar_tasks_returns_empty_dict():
    _fake_tm.result = []
    assert classify_prompt("anything") == {}


def test_single_task_does_not_divide_by_zero():
    _fake_tm.result = [("coding", 0.83)]
    assert classify_prompt("write code") == {"coding": 1.0}


def test_all_identical_scores_do_not_divide_by_zero():
    _fake_tm.result = [("a", 0.7), ("b", 0.7), ("c", 0.7)]
    out = classify_prompt("q")
    assert out == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_spread_scores_normalize_and_threshold_at_half():
    _fake_tm.result = [("a", 0.9), ("b", 0.5), ("c", 0.1)]
    out = classify_prompt("q")
    assert out == {"a": 1.0, "b": 0.5}  # c normalizes to 0.0, below the 0.5 cut


def test_scores_are_rounded_to_four_places():
    _fake_tm.result = [("a", 1.0), ("b", 0.0), ("x", 0.66666)]
    out = classify_prompt("q")
    assert out["x"] == pytest.approx(0.6667, abs=1e-9)
