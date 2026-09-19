"""Synthetic client, portfolio and market data.

Three datasets, each generated with the structure the corresponding model is
supposed to find:

- `applicants`  - credit applications with a default outcome driven by a latent
  risk score, so a well-specified model can recover the relationship and a
  badly-specified one cannot hide.
- `portfolio`   - outstanding exposures with segment and sector concentration,
  so the risk layer has something to concentrate on.
- `market`      - daily series with trend, annual seasonality, weekday effects
  and volatility clustering, which is what makes the ARIMA/LSTM comparison
  meaningful rather than decorative.

Run with:  python -m data.generate
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "raw"
SEED = 20260919

N_APPLICANTS = 24_000
N_EXPOSURES = 6_000
MARKET_DAYS = 1_460          # four years of daily observations
MARKET_START_VALUE = 250_000_000.0   # portfolio mark-to-market, in EUR

SEGMENTS = ["retail", "sme", "corporate"]
SECTORS = ["manufacturing", "retail_trade", "construction", "services",
           "transport", "agriculture", "technology"]
REGIONS = ["north", "south", "east", "west", "central"]
PURPOSES = ["working_capital", "equipment", "property", "refinance", "consumer"]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_applicants(rng: np.random.Generator) -> pd.DataFrame:
    n = N_APPLICANTS

    # Application dates come first, because the population drifts over them.
    # Underwriting standards loosened across the period: more recent applicants
    # carry more debt relative to income and score slightly lower at the bureau.
    # That drift is what the validation layer exists to detect, and without it a
    # PSI report is a page of zeros that proves nothing.
    days_ago = rng.integers(30, 1_095, n)
    recency = 1.0 - (days_ago - 30) / (1_095 - 30)      # 0 = oldest, 1 = newest

    age = np.clip(rng.normal(44, 13, n), 19, 82).round(0)
    income = np.clip(rng.lognormal(10.6, 0.55, n), 9_000, 600_000).round(0)
    employment_years = np.clip(rng.gamma(2.6, 2.7, n), 0, 45).round(1)
    existing_loans = rng.poisson(1.3, n)
    loan_amount = np.clip(income * rng.uniform(0.15, 2.4, n), 1_500, 900_000).round(0)
    term_months = rng.choice([12, 24, 36, 48, 60, 84, 120], n,
                             p=[0.09, 0.16, 0.24, 0.19, 0.16, 0.10, 0.06])

    debt_to_income = np.clip(
        (loan_amount / term_months * 12) / income
        + rng.normal(0, 0.05, n)
        + 0.22 * recency,                               # the loosening, over time
        0.01, 3.5,
    )
    # Bureau score with a realistic distribution and a genuine relationship to
    # the drivers above, so it is informative without being the whole answer.
    bureau_score = np.clip(
        620
        + 42 * np.log1p(employment_years)
        - 78 * debt_to_income
        + 0.00012 * income
        - 9 * existing_loans
        - 18 * recency                                  # bureau scores drift down
        + rng.normal(0, 46, n),
        300, 900,
    ).round(0)

    segment = rng.choice(SEGMENTS, n, p=[0.62, 0.29, 0.09])
    sector = rng.choice(SECTORS, n)
    region = rng.choice(REGIONS, n)
    purpose = rng.choice(PURPOSES, n)

    previous_default = (rng.random(n) < _sigmoid(-3.1 + 1.9 * debt_to_income)).astype(int)
    months_since_last_delinquency = np.where(
        rng.random(n) < 0.34, np.clip(rng.exponential(19, n), 0, 120).round(0), -1
    )

    # --- latent risk, and the outcome -------------------------------------
    sector_effect = pd.Series(sector).map({
        "construction": 0.42, "transport": 0.26, "retail_trade": 0.18,
        "agriculture": 0.30, "manufacturing": 0.05, "services": -0.02,
        "technology": -0.24,
    }).to_numpy()
    segment_effect = pd.Series(segment).map(
        {"retail": 0.0, "sme": 0.28, "corporate": -0.32}
    ).to_numpy()

    logit = (
        -3.05
        - 0.0069 * (bureau_score - 620)
        + 1.35 * debt_to_income
        + 0.92 * previous_default
        - 0.055 * employment_years
        + 0.17 * existing_loans
        + sector_effect
        + segment_effect
        + 0.0028 * (term_months - 36)
        # A deliberate non-linearity: very young and very old applicants are both
        # riskier. A plain linear-in-age model cannot represent this, which is
        # what the binned scorecard is there to handle.
        + 0.00055 * (age - 45) ** 2
        + rng.normal(0, 0.55, n)
    )
    default_probability = _sigmoid(logit)
    defaulted = (rng.random(n) < default_probability).astype(int)

    application_date = [
        dt.date.today() - dt.timedelta(days=int(offset)) for offset in days_ago
    ]

    return pd.DataFrame({
        "application_id": [f"APP-{index:06d}" for index in range(n)],
        "application_date": application_date,
        "age": age,
        "annual_income": income,
        "employment_years": employment_years,
        "existing_loans": existing_loans,
        "loan_amount": loan_amount,
        "term_months": term_months,
        "debt_to_income": debt_to_income.round(4),
        "bureau_score": bureau_score,
        "segment": segment,
        "sector": sector,
        "region": region,
        "purpose": purpose,
        "previous_default": previous_default,
        "months_since_last_delinquency": months_since_last_delinquency,
        "defaulted": defaulted,
    })


def generate_portfolio(rng: np.random.Generator, applicants: pd.DataFrame) -> pd.DataFrame:
    """Outstanding exposures, deliberately concentrated.

    A portfolio spread evenly across sectors has no concentration risk to find,
    so construction and retail trade are over-weighted on purpose.
    """
    booked = applicants.sample(n=N_EXPOSURES, random_state=SEED).reset_index(drop=True)

    sector_weight = booked["sector"].map({
        "construction": 2.4, "retail_trade": 2.0, "manufacturing": 1.0,
        "services": 0.9, "transport": 1.1, "agriculture": 0.7, "technology": 0.6,
    }).to_numpy()

    exposure = (booked["loan_amount"].to_numpy() * rng.uniform(0.35, 1.0, N_EXPOSURES)
                * sector_weight).round(0)

    # Loss given default depends on collateral, which depends on purpose.
    lgd = np.clip(
        booked["purpose"].map({
            "property": 0.24, "equipment": 0.38, "working_capital": 0.58,
            "refinance": 0.47, "consumer": 0.66,
        }).to_numpy() + rng.normal(0, 0.07, N_EXPOSURES),
        0.05, 0.95,
    )

    return pd.DataFrame({
        "exposure_id": [f"EXP-{index:06d}" for index in range(N_EXPOSURES)],
        "application_id": booked["application_id"],
        "client_segment": booked["segment"],
        "sector": booked["sector"],
        "region": booked["region"],
        "purpose": booked["purpose"],
        "exposure_at_default": exposure,
        "loss_given_default": lgd.round(4),
        "remaining_term_months": np.clip(
            booked["term_months"] - rng.integers(0, 36, N_EXPOSURES), 1, None
        ),
        "days_past_due": np.where(
            rng.random(N_EXPOSURES) < 0.11,
            rng.integers(1, 180, N_EXPOSURES), 0
        ),
    })


def generate_market(rng: np.random.Generator) -> pd.DataFrame:
    """A daily mark-to-market portfolio value with structure worth forecasting.

    Two properties are deliberate. The series carries a trend, an annual cycle
    and a weekday effect, which is what the ARIMA and LSTM comparison needs. And
    the returns are genuinely fat-tailed - volatility clusters and occasional
    jumps land on top of it - which is what makes the historical and parametric
    VaR disagree. A Gaussian return series would make the parametric figure
    correct and the whole comparison pointless.
    """
    dates = [dt.date.today() - dt.timedelta(days=MARKET_DAYS - index)
             for index in range(MARKET_DAYS)]

    level = MARKET_START_VALUE
    values, returns = [], []
    volatility = 0.010

    for day in dates:
        trend = 0.00022
        annual = 0.0016 * math.sin(2 * math.pi * (day.timetuple().tm_yday - 40) / 365)
        weekday = {0: -0.0004, 1: 0.0003, 2: 0.0004, 3: 0.0002, 4: -0.0006,
                   5: 0.0, 6: 0.0}[day.weekday()]

        # GARCH-like clustering: today's volatility remembers yesterday's shock.
        # The coefficients must satisfy beta + alpha * E|z| < 1 or the recursion
        # is explosive and volatility simply pins to its upper clip - which looks
        # like a working model and is a broken one. Here
        # 0.85 + 0.15 * sqrt(2/pi) = 0.97, giving a long-run daily volatility of
        # omega / (1 - 0.97) = 1.1%, or roughly 17% annualised.
        shock = rng.normal(0, volatility)
        volatility = float(
            np.clip(0.00033 + 0.85 * volatility + 0.15 * abs(shock), 0.004, 0.05)
        )

        # Jumps: rare, large, and the reason a normal distribution is the wrong
        # model for the tail. Roughly four a year, at about 3.5 standard
        # deviations each.
        jump = rng.normal(0, 0.042) if rng.random() < 0.012 else 0.0

        daily_return = trend + annual + weekday + shock + jump
        level *= (1 + daily_return)
        values.append(level)
        returns.append(daily_return)

    return pd.DataFrame({
        "observation_date": dates,
        "portfolio_value": np.round(values, 2),
        "daily_return": np.round(returns, 6),
    })


def generate(seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    applicants = generate_applicants(rng)
    portfolio = generate_portfolio(rng, applicants)
    market = generate_market(rng)

    applicants.to_parquet(RAW_DIR / "applicants.parquet", index=False)
    portfolio.to_parquet(RAW_DIR / "portfolio.parquet", index=False)
    market.to_parquet(RAW_DIR / "market.parquet", index=False)

    return {
        "applicants": len(applicants),
        "default_rate_pct": round(float(applicants["defaulted"].mean()) * 100, 2),
        "exposures": len(portfolio),
        "total_exposure_eur": int(portfolio["exposure_at_default"].sum()),
        "market_days": len(market),
    }


if __name__ == "__main__":
    import json

    print("Generating finance datasets ...")
    print(json.dumps(generate(), indent=2))
    print(f"\nWritten to {RAW_DIR}")
