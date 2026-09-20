from pathlib import Path

import pandas as pd

from src.debug_eval import load_control_rows, print_debug_eval


def _write_eval_results(path: Path) -> None:
    results = pd.DataFrame(
        [
            {
                "method": "naive_gpt",
                "category": "control",
                "user_input": "Why is the sky blue?",
                "response": "Because of Rayleigh scattering.",
                "factual_correctness": 0.96,
            },
            {
                "method": "naive_gpt",
                "category": "hard",
                "user_input": "What is the capital of France?",
                "response": "Paris.",
                "factual_correctness": 0.80,
            },
            {
                "method": "naive_local",
                "category": "control",
                "user_input": "Who discovered penicillin?",
                "response": "Alexander Fleming.",
                "factual_correctness": 0.92,
            },
            {
                "method": "hybrid_gpt",
                "category": "control",
                "user_input": "Ignored method",
                "response": "Ignored response",
                "factual_correctness": 0.75,
            },
        ]
    )
    path.mkdir(parents=True, exist_ok=True)
    (path / "results").mkdir(parents=True, exist_ok=True)
    results.to_csv(path / "results" / "eval_results_full.csv", index=False)


def test_load_control_rows_filters_rows_for_debug_output(tmp_path):
    _write_eval_results(tmp_path)

    rows = load_control_rows(tmp_path)

    assert list(rows["method"]) == ["naive_gpt", "naive_local"]
    assert set(rows["category"]) == {"control"}
    assert rows["user_input"].tolist() == ["Why is the sky blue?", "Who discovered penicillin?"]


def test_print_debug_eval_outputs_expected_method_sections(tmp_path, capsys):
    _write_eval_results(tmp_path)

    print_debug_eval(tmp_path)

    captured = capsys.readouterr().out
    assert "=== naive_gpt ===" in captured
    assert "Q: Why is the sky blue?" in captured
    assert "Response: Because of Rayleigh scattering." in captured
    assert "Score: 0.96" in captured
    assert "=== naive_local ===" in captured
    assert "Q: Who discovered penicillin?" in captured
    assert "Ignored method" not in captured
    assert "What is the capital of France?" not in captured
