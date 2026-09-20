from pathlib import Path

import pandas as pd

DEFAULT_METHODS = ["naive_gpt", "naive_local"]


def load_control_rows(project_root: Path, methods: list[str] | tuple[str, ...] = DEFAULT_METHODS) -> pd.DataFrame:
    df = pd.read_csv(project_root / "results" / "eval_results_full.csv")
    return df[
        df["method"].isin(methods) & (df["category"] == "control")
    ].copy()


def print_debug_eval(project_root: Path, methods: list[str] | tuple[str, ...] = DEFAULT_METHODS) -> None:
    df = load_control_rows(project_root, methods)

    for method in methods:
        subset = df[df["method"] == method]
        print(f"=== {method} ===")
        for _, row in subset.iterrows():
            print(f"Q: {row['user_input']}")
            print(f"Response: {row['response']}")
            print(f"Score: {row['factual_correctness']}")
            print("---")


def main() -> None:
    project_root = Path(__file__).parent.parent
    print_debug_eval(project_root)


if __name__ == "__main__":
    main()