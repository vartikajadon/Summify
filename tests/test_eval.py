import pytest
from eval.metrics import run_evaluation

def test_eval_metrics_harness(tmp_path):
    """
    Verifies that run_evaluation executes all 6 metric benchmarks and returns True (all pass).
    """
    fixtures_dir = str(tmp_path / "eval_fixtures")
    passed = run_evaluation(fixtures_path=fixtures_dir)
    assert passed is True
