"""Portfolio risk: expected loss, concentration, and value at risk.

Expected loss is the straightforward part:

    EL = PD x LGD x EAD

and it is straightforward only if PD is a calibrated probability. The scoring
module reports a calibration table for exactly this reason: a model that ranks
perfectly but predicts 8% where the truth is 12% produces an expected loss that
is 50% too low across the whole book, and nothing about the AUC would have
warned anyone.

Concentration is measured rather than eyeballed. A Herfindahl-Hirschman Index
over sector exposure turns "we feel heavy in construction" into a number that can
be tracked and breached.

VaR is computed two ways on purpose - historical and parametric - because the
gap between them is the point. The return series has volatility clustering, so
the normal assumption understates the tail, and reporting only the parametric
number is how a risk report becomes reassuring and wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

# Concentration thresholds. Conventional readings, stated so the output is
# interpretable without a reference text to hand.
HHI_BANDS = [(0.15, "diversified"), (0.25, "moderately concentrated"), (1.01, "concentrated")]


def _load(name: str, directory: Path = RAW_DIR) -> pd.DataFrame:
    path = directory / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `python -m data.generate` and "
            f"`python -m models.credit_scoring` first."
        )
    return pd.read_parquet(path)


def expected_loss() -> pd.DataFrame:
    """Join calibrated PDs onto the exposure book and compute EL per exposure."""
    portfolio = _load("portfolio")
    scored = _load("scored_applicants", OUTPUT_DIR)

    book = portfolio.merge(
        scored[["application_id", "probability_of_default", "score_points"]],
        on="application_id",
        how="inner",
    )
    if book.empty:
        raise RuntimeError(
            "No exposures could be scored. The scoring model is trained on a "
            "time split, so only applications in the held-out period have a PD."
        )

    book["expected_loss"] = (
        book["probability_of_default"]
        * book["loss_given_default"]
        * book["exposure_at_default"]
    ).round(2)
    book["loss_rate_pct"] = (
        book["expected_loss"] / book["exposure_at_default"] * 100
    ).round(3)
    # Past-due exposures are not automatically defaults, but they are where a
    # review starts, so they are flagged rather than re-rated.
    book["watchlist"] = (book["days_past_due"] >= 90) | (book["probability_of_default"] >= 0.35)
    return book


def herfindahl(shares: pd.Series) -> float:
    """HHI on exposure shares. 1/n is perfectly diversified, 1.0 is one name."""
    weights = shares / shares.sum()
    return float((weights ** 2).sum())


def concentration(book: pd.DataFrame) -> dict:
    result = {}
    for dimension in ("sector", "region", "client_segment"):
        grouped = book.groupby(dimension).agg(
            exposure=("exposure_at_default", "sum"),
            expected_loss=("expected_loss", "sum"),
            exposures=("exposure_id", "count"),
        )
        grouped["exposure_share_pct"] = (
            grouped["exposure"] / grouped["exposure"].sum() * 100
        ).round(2)
        grouped["loss_share_pct"] = (
            grouped["expected_loss"] / grouped["expected_loss"].sum() * 100
        ).round(2)
        # Where loss share exceeds exposure share, that slice of the book is
        # riskier than its size suggests. That ratio is the finding, not the
        # exposure number on its own.
        grouped["risk_intensity"] = (
            grouped["loss_share_pct"] / grouped["exposure_share_pct"]
        ).round(2)

        index = herfindahl(grouped["exposure"])
        band = next(label for threshold, label in HHI_BANDS if index < threshold)

        result[dimension] = {
            "hhi": round(index, 4),
            "assessment": band,
            "effective_number_of_groups": round(1 / index, 1),
            "breakdown": grouped.sort_values("exposure", ascending=False)
            .reset_index()
            .to_dict(orient="records"),
        }
    return result


def value_at_risk(confidence: float = 0.99, horizon_days: int = 1) -> dict:
    """Historical and parametric VaR on the portfolio return series."""
    market = _load("market")
    returns = market["daily_return"].to_numpy(dtype=float)
    current_value = float(market["portfolio_value"].iloc[-1])

    # Historical: the empirical quantile. Makes no distributional assumption and
    # therefore keeps the fat tail the data actually has.
    historical_quantile = float(np.percentile(returns, (1 - confidence) * 100))
    historical_var = -historical_quantile * current_value * np.sqrt(horizon_days)

    # Parametric: normal assumption. Reported so the understatement is visible.
    mean, sigma = float(np.mean(returns)), float(np.std(returns, ddof=1))
    z = float(stats.norm.ppf(1 - confidence))
    parametric_var = -(mean + z * sigma) * current_value * np.sqrt(horizon_days)

    # Expected shortfall: the average loss given that VaR is breached. The
    # number that says how bad the bad days are, which VaR by construction does
    # not.
    tail = returns[returns <= historical_quantile]
    expected_shortfall = -float(tail.mean()) * current_value * np.sqrt(horizon_days)

    excess_kurtosis = float(stats.kurtosis(returns))
    jarque_bera = stats.jarque_bera(returns)

    return {
        "confidence_pct": confidence * 100,
        "horizon_days": horizon_days,
        "portfolio_value": round(current_value, 2),
        "historical_var": round(historical_var, 2),
        "parametric_var_normal": round(parametric_var, 2),
        "expected_shortfall": round(expected_shortfall, 2),
        "parametric_understatement_pct": round(
            (historical_var / parametric_var - 1) * 100, 2
        ),
        "observations": len(returns),
        "annualised_volatility_pct": round(sigma * np.sqrt(252) * 100, 2),
        "excess_kurtosis": round(excess_kurtosis, 3),
        "normality_rejected": bool(jarque_bera.pvalue < 0.01),
        "note": (
            "Returns are not normal (excess kurtosis above zero, Jarque-Bera "
            "rejects). The parametric figure is reported for comparison only; "
            "plan against the historical VaR and the expected shortfall."
        ),
    }


def stress_test(book: pd.DataFrame, pd_multipliers: dict[str, float] | None = None) -> list[dict]:
    """What expected loss becomes under adverse scenarios.

    A single expected loss figure answers "what do we expect". A board asks "what
    if we are wrong", and the answer has to be computed the same way.
    """
    scenarios = pd_multipliers or {
        "base": 1.0,
        "mild_downturn": 1.4,
        "severe_downturn": 2.2,
        "sector_shock_construction": 1.0,   # handled below
    }

    rows = []
    baseline = float(book["expected_loss"].sum())

    for name, multiplier in scenarios.items():
        stressed = book.copy()
        if name == "sector_shock_construction":
            mask = stressed["sector"] == "construction"
            stressed.loc[mask, "probability_of_default"] *= 3.0
            # A downturn raises loss given default as well as default rates:
            # collateral is worth less precisely when it is being realised.
            stressed.loc[mask, "loss_given_default"] = np.clip(
                stressed.loc[mask, "loss_given_default"] * 1.25, 0, 0.95
            )
        else:
            stressed["probability_of_default"] = np.clip(
                stressed["probability_of_default"] * multiplier, 0, 1
            )

        loss = float(
            (stressed["probability_of_default"]
             * stressed["loss_given_default"]
             * stressed["exposure_at_default"]).sum()
        )
        rows.append({
            "scenario": name,
            "expected_loss": round(loss, 2),
            "change_vs_base_pct": round((loss / baseline - 1) * 100, 2),
            "loss_rate_pct": round(loss / float(book["exposure_at_default"].sum()) * 100, 3),
        })
    return rows


def run(save: bool = True) -> dict:
    book = expected_loss()

    total_exposure = float(book["exposure_at_default"].sum())
    total_expected_loss = float(book["expected_loss"].sum())

    report = {
        "portfolio": {
            "exposures": int(len(book)),
            "total_exposure_eur": round(total_exposure, 2),
            "total_expected_loss_eur": round(total_expected_loss, 2),
            "portfolio_loss_rate_pct": round(total_expected_loss / total_exposure * 100, 3),
            "exposure_weighted_pd_pct": round(
                float((book["probability_of_default"] * book["exposure_at_default"]).sum()
                      / total_exposure) * 100, 3
            ),
            "exposure_weighted_lgd_pct": round(
                float((book["loss_given_default"] * book["exposure_at_default"]).sum()
                      / total_exposure) * 100, 2
            ),
            "watchlist_exposures": int(book["watchlist"].sum()),
            "watchlist_exposure_eur": round(
                float(book.loc[book["watchlist"], "exposure_at_default"].sum()), 2
            ),
        },
        "concentration": concentration(book),
        "value_at_risk": value_at_risk(),
        "stress_tests": stress_test(book),
        "largest_expected_losses": book.nlargest(10, "expected_loss")[
            ["exposure_id", "sector", "client_segment", "exposure_at_default",
             "probability_of_default", "loss_given_default", "expected_loss"]
        ].round(4).to_dict(orient="records"),
    }

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "risk_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        book.to_parquet(OUTPUT_DIR / "exposure_risk.parquet", index=False)

    return report


if __name__ == "__main__":
    report = run()
    portfolio = report["portfolio"]

    print("Portfolio risk")
    print(f"  Exposures              {portfolio['exposures']:,}")
    print(f"  Total exposure         EUR {portfolio['total_exposure_eur']:,.0f}")
    print(f"  Expected loss          EUR {portfolio['total_expected_loss_eur']:,.0f} "
          f"({portfolio['portfolio_loss_rate_pct']}% of exposure)")
    print(f"  Exposure-weighted PD   {portfolio['exposure_weighted_pd_pct']}%")
    print(f"  Watchlist              {portfolio['watchlist_exposures']} exposures, "
          f"EUR {portfolio['watchlist_exposure_eur']:,.0f}")

    sector = report["concentration"]["sector"]
    print(f"\nSector concentration: HHI {sector['hhi']} ({sector['assessment']}, "
          f"effectively {sector['effective_number_of_groups']} sectors)")
    for row in sector["breakdown"][:5]:
        print(f"  {row['sector']:<16} {row['exposure_share_pct']:>5.1f}% of exposure, "
              f"{row['loss_share_pct']:>5.1f}% of loss, intensity {row['risk_intensity']}")

    var = report["value_at_risk"]
    print(f"\n1-day 99% VaR")
    print(f"  Historical             EUR {var['historical_var']:,.0f}")
    print(f"  Parametric (normal)    EUR {var['parametric_var_normal']:,.0f}  "
          f"(understates by {var['parametric_understatement_pct']}%)")
    print(f"  Expected shortfall     EUR {var['expected_shortfall']:,.0f}")
    print(f"  Excess kurtosis {var['excess_kurtosis']}, normality rejected: "
          f"{var['normality_rejected']}")

    print("\nStress tests")
    for row in report["stress_tests"]:
        print(f"  {row['scenario']:<30} EUR {row['expected_loss']:>14,.0f}  "
              f"{row['change_vs_base_pct']:+.1f}%")
