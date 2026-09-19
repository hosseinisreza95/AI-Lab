"""CLI for the finance risk and forecasting pipeline.

    python main.py generate                     # synthetic applicants, portfolio, market
    python main.py score                        # credit scoring: scorecard vs challenger
    python main.py risk                         # expected loss, concentration, VaR, stress
    python main.py forecast                     # SARIMA vs LSTM vs baselines
    python main.py validate                     # population stability and feature drift
    python main.py explain APP-001234           # why one applicant got their score
    python main.py all                          # the whole pipeline in order
"""
from __future__ import annotations

import argparse
import json
import sys


def cmd_generate(args: argparse.Namespace) -> int:
    from data.generate import generate

    print("Generating datasets ...")
    print(json.dumps(generate(), indent=2))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    from models.credit_scoring import train

    report = train()
    print("Credit scoring, held-out period "
          f"{report['test_period'][0]} to {report['test_period'][1]} "
          f"({report['test_rows']:,} applications)")
    print(f"  Scorecard (WOE + logistic)  {report['scorecard']}")
    print(f"  Challenger (calibrated GBM) {report['challenger_gbm']}")
    print(f"  Gini cost of interpretability: "
          f"{report['gini_cost_of_interpretability']:+.4f}")

    reversals = [row for row in report["scorecard_coefficients"]
                 if row["direction"] != "as expected"]
    if reversals:
        print(f"\n  {len(reversals)} coefficient sign reversal(s) - investigate before use:")
        for row in reversals:
            print(f"    {row['feature']} ({row['coefficient']})")
    else:
        print("\n  All coefficient signs as expected.")
    return 0


def cmd_risk(args: argparse.Namespace) -> int:
    from models.risk_analysis import run

    report = run()
    portfolio, var = report["portfolio"], report["value_at_risk"]

    print(f"Portfolio: {portfolio['exposures']:,} exposures, "
          f"EUR {portfolio['total_exposure_eur']:,.0f}")
    print(f"  Expected loss  EUR {portfolio['total_expected_loss_eur']:,.0f} "
          f"({portfolio['portfolio_loss_rate_pct']}%)")
    print(f"  1-day 99% VaR  EUR {var['historical_var']:,.0f} historical, "
          f"EUR {var['parametric_var_normal']:,.0f} parametric "
          f"(normal understates by {var['parametric_understatement_pct']}%)")
    print(f"  Expected shortfall EUR {var['expected_shortfall']:,.0f}")

    sector = report["concentration"]["sector"]
    print(f"  Sector HHI {sector['hhi']} ({sector['assessment']})")
    for row in report["stress_tests"]:
        print(f"    {row['scenario']:<30} EUR {row['expected_loss']:>14,.0f}  "
              f"{row['change_vs_base_pct']:+.1f}%")
    return 0


def cmd_forecast(args: argparse.Namespace) -> int:
    from models.forecasting import run

    report = run(horizon=args.horizon, folds=args.folds)
    print(f"Rolling-origin backtest: {report['folds_completed']} folds, "
          f"{report['horizon_days']}-day horizon")
    print(f"{'model':<16}{'MAE':>14}{'MAPE %':>10}{'direction %':>14}")
    for row in report["results"]:
        direction = ("n/a" if row["directional_accuracy_pct"] is None
                     else f"{row['directional_accuracy_pct']:.1f}")
        print(f"{row['model']:<16}{row['mae']:>14,.0f}"
              f"{row['mape_pct']:>10.3f}{direction:>14}")
    print(f"\nBest: {report['best_model']} | beats naive: {report['beats_naive']} | "
          f"SARIMA converged {report['sarima_convergence']['converged_fits']}/"
          f"{report['sarima_convergence']['total_fits']}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from validation.stability import run

    report = run()
    print(f"Score PSI {report['score_psi']} ({report['score_assessment']})")
    rate = report["observed_default_rate"]
    print(f"Observed default rate {rate['reference_pct']}% -> {rate['current_pct']}%")
    print("\nFeature drift:")
    for row in report["feature_drift"]:
        marker = "  <-- " if row["assessment"] != "stable" else "      "
        print(f"  {row['feature']:<32} PSI {row['psi']:<10}{marker}{row['assessment']}")

    unstable = [row for row in report["feature_drift"] if row["assessment"] != "stable"]
    if unstable:
        print(f"\n{len(unstable)} feature(s) outside the stable band. An aggregate "
              "score PSI can look calm while an input has moved materially - that "
              "is why both are reported.")
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    from features.engineering import load_applicants
    from models.credit_scoring import load_scorecard

    applicants = load_applicants()
    match = applicants[applicants["application_id"] == args.application_id]
    if match.empty:
        print(f"No application '{args.application_id}'. "
              f"Try one of: {', '.join(applicants['application_id'].head(3))}")
        return 1

    row = match.iloc[0]
    explanation = load_scorecard().explain(row)

    print(f"{args.application_id}")
    print(f"  Score            {explanation['score_points']} points")
    print(f"  Probability of default  {explanation['probability_of_default_pct']}%")
    print(f"  Actual outcome   {'DEFAULTED' if row['defaulted'] else 'performed'}")
    print("\n  Largest contributions to risk:")
    for reason in explanation["top_risk_drivers"]:
        print(f"    {reason['feature']:<30} {str(reason['bin']):<28} "
              f"{reason['contribution_to_risk']:+.4f}")
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for name, function in (
        ("GENERATE", cmd_generate), ("SCORE", cmd_score), ("RISK", cmd_risk),
        ("FORECAST", cmd_forecast), ("VALIDATE", cmd_validate),
    ):
        print(f"\n{'=' * 72}\n{name}\n{'=' * 72}")
        code = function(args)
        if code:
            return code
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Finance: credit scoring, risk analysis and time-series forecasting.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("generate", help="Generate the synthetic datasets").set_defaults(
        func=cmd_generate
    )
    subparsers.add_parser("score", help="Train and evaluate the credit models").set_defaults(
        func=cmd_score
    )
    subparsers.add_parser("risk", help="Expected loss, concentration, VaR, stress").set_defaults(
        func=cmd_risk
    )

    forecast = subparsers.add_parser("forecast", help="Backtest SARIMA, LSTM and baselines")
    forecast.add_argument("--horizon", type=int, default=14)
    forecast.add_argument("--folds", type=int, default=6)
    forecast.set_defaults(func=cmd_forecast)

    subparsers.add_parser("validate", help="Population stability and feature drift").set_defaults(
        func=cmd_validate
    )

    explain = subparsers.add_parser("explain", help="Explain one applicant's score")
    explain.add_argument("application_id", type=str)
    explain.set_defaults(func=cmd_explain)

    everything = subparsers.add_parser("all", help="Run the whole pipeline")
    everything.add_argument("--horizon", type=int, default=14)
    everything.add_argument("--folds", type=int, default=6)
    everything.set_defaults(func=cmd_all)

    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    sys.exit(parsed.func(parsed))
