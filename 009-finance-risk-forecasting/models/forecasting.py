"""Time-series forecasting: SARIMA, an LSTM, and the baselines both must beat.

The brief for this work was to choose per problem, depending on data volume,
seasonality and interpretability requirements. That is only a real choice if both
are implemented and evaluated the same way, so both are here and the walk-forward
backtest picks.

Three baselines, because a forecasting result without one is not a result:

- **naive** - tomorrow equals today. On a near-random-walk series this is
  genuinely hard to beat, and any model that cannot is adding risk for nothing.
- **drift** - naive plus the average historical change.
- **seasonal naive** - the value one week ago, which captures the weekday effect.

Evaluation is rolling-origin: fit on everything up to a cut point, forecast the
next `horizon` days, move the cut point forward, repeat. Every fold is a genuine
out-of-sample forecast made with no knowledge of its own future.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "outputs"

warnings.filterwarnings("ignore")

SEASONAL_PERIOD = 7
# (0,1,1) is a random walk with an MA error term, plus a weekly seasonal pair.
# A richer (2,1,2) fitted here converged in only 2 of 6 backtest folds and scored
# worse than the naive baseline as a direct result - the method was not losing,
# the optimiser was. Parsimony is not a stylistic preference on a series this
# close to a random walk; it is what makes the fit reproducible.
DEFAULT_ORDER = (0, 1, 1)
DEFAULT_SEASONAL_ORDER = (1, 0, 1, SEASONAL_PERIOD)

LSTM_LOOKBACK = 28
LSTM_HIDDEN = 48
LSTM_EPOCHS = 45
LSTM_LR = 0.01


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class FoldResult:
    model: str
    mae: float
    rmse: float
    mape: float
    directional_accuracy: float


def load_series() -> pd.Series:
    path = RAW_DIR / "market.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing. Run `python -m data.generate` first.")
    frame = pd.read_parquet(path)
    frame["observation_date"] = pd.to_datetime(frame["observation_date"])
    return frame.set_index("observation_date")["portfolio_value"].astype(float)


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------

def forecast_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    return np.repeat(history[-1], horizon)


def forecast_drift(history: np.ndarray, horizon: int) -> np.ndarray:
    slope = (history[-1] - history[0]) / max(len(history) - 1, 1)
    return history[-1] + slope * np.arange(1, horizon + 1)


def forecast_seasonal_naive(history: np.ndarray, horizon: int) -> np.ndarray:
    if len(history) < SEASONAL_PERIOD:
        return forecast_naive(history, horizon)
    window = history[-SEASONAL_PERIOD:]
    return np.array([window[index % SEASONAL_PERIOD] for index in range(horizon)])


# ---------------------------------------------------------------------------
# SARIMA
# ---------------------------------------------------------------------------

# Populated by forecast_sarima so the backtest can report how often the
# likelihood optimisation actually converged. A forecast from a fit that did not
# converge is not a verdict on the model, and reporting one as if it were is how
# a method gets discarded for the wrong reason.
SARIMA_CONVERGENCE: list[bool] = []


def forecast_sarima(history: np.ndarray, horizon: int) -> np.ndarray:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        history,
        order=DEFAULT_ORDER,
        seasonal_order=DEFAULT_SEASONAL_ORDER,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False)
    SARIMA_CONVERGENCE.append(bool(fitted.mle_retvals.get("converged", False)))
    return np.asarray(fitted.forecast(steps=horizon), dtype=float)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

def forecast_lstm(history: np.ndarray, horizon: int, seed: int = 42) -> np.ndarray:
    """A small LSTM over differenced, standardised windows.

    Trained on log returns rather than levels. Handing a network a
    non-stationary price series and asking for the next price teaches it to copy
    the last value, which is the naive baseline with a GPU bill attached.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)

    log_levels = np.log(history)
    returns = np.diff(log_levels)
    if len(returns) <= LSTM_LOOKBACK + 10:
        return forecast_naive(history, horizon)

    mean, std = returns.mean(), returns.std() + 1e-9
    scaled = (returns - mean) / std

    windows = np.lib.stride_tricks.sliding_window_view(scaled, LSTM_LOOKBACK + 1)
    x = torch.tensor(windows[:, :-1], dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(windows[:, -1], dtype=torch.float32).unsqueeze(-1)

    class Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(1, LSTM_HIDDEN, batch_first=True)
            self.head = nn.Linear(LSTM_HIDDEN, 1)

        def forward(self, batch):
            output, _ = self.lstm(batch)
            return self.head(output[:, -1, :])

    net = Net()
    optimiser = torch.optim.Adam(net.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    net.train()
    for _ in range(LSTM_EPOCHS):
        optimiser.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        optimiser.step()

    # Recursive multi-step: each prediction is fed back in as the next input.
    net.eval()
    window = scaled[-LSTM_LOOKBACK:].copy()
    level = log_levels[-1]
    predictions = []
    with torch.no_grad():
        for _ in range(horizon):
            batch = torch.tensor(window, dtype=torch.float32).reshape(1, LSTM_LOOKBACK, 1)
            step = float(net(batch).item())
            window = np.append(window[1:], step)
            level += step * std + mean
            predictions.append(float(np.exp(level)))

    return np.array(predictions)


# ---------------------------------------------------------------------------
# backtesting
# ---------------------------------------------------------------------------

def _metrics(name: str, actual: np.ndarray, predicted: np.ndarray,
             last_observed: float) -> FoldResult:
    errors = np.abs(actual - predicted)
    # Did the forecast get the direction of travel right? On a financial series
    # this often matters more than the level, and a model can have a good MAE
    # while being wrong about direction every single time.
    actual_direction = np.sign(actual - last_observed)
    predicted_direction = np.sign(predicted - last_observed)

    # A flat forecast (the naive baseline) makes no directional call at all.
    # Scoring those steps as wrong reports naive at 0% directional accuracy,
    # which reads as "always wrong" when the truth is "never asked".
    called = predicted_direction != 0
    directional = (
        float((actual_direction[called] == predicted_direction[called]).mean() * 100)
        if called.any() else float("nan")
    )

    return FoldResult(
        model=name,
        mae=float(errors.mean()),
        rmse=float(np.sqrt(((actual - predicted) ** 2).mean())),
        mape=float((errors / np.abs(actual)).mean() * 100),
        directional_accuracy=directional,
    )


def backtest(horizon: int = 14, folds: int = 6, include_lstm: bool = True) -> dict:
    series = load_series()
    values = series.to_numpy(dtype=float)

    models = {
        "naive": forecast_naive,
        "drift": forecast_drift,
        "seasonal_naive": forecast_seasonal_naive,
        "sarima": forecast_sarima,
    }
    if include_lstm and torch_available():
        models["lstm"] = forecast_lstm

    collected: dict[str, list[FoldResult]] = {name: [] for name in models}
    SARIMA_CONVERGENCE.clear()
    completed = 0

    for fold in range(folds):
        cut = len(values) - horizon * (folds - fold)
        if cut < 200:
            continue

        history = values[:cut]
        actual = values[cut:cut + horizon]
        if len(actual) < horizon:
            continue

        for name, function in models.items():
            try:
                predicted = function(history, horizon)
            except Exception as exc:
                print(f"  ! {name} failed on fold {fold}: {exc}")
                continue
            collected[name].append(_metrics(name, actual, predicted, history[-1]))
        completed += 1

    summary = []
    for name, results in collected.items():
        if not results:
            continue
        summary.append({
            "model": name,
            "mae": round(float(np.mean([r.mae for r in results])), 2),
            "rmse": round(float(np.mean([r.rmse for r in results])), 2),
            "mape_pct": round(float(np.mean([r.mape for r in results])), 3),
            "directional_accuracy_pct": (
                None
                if np.isnan(np.nanmean([r.directional_accuracy for r in results]))
                else round(float(np.nanmean([r.directional_accuracy for r in results])), 1)
            ),
            "folds": len(results),
        })
    summary.sort(key=lambda row: row["mape_pct"])

    best = summary[0]["model"] if summary else None
    naive = next((row for row in summary if row["model"] == "naive"), None)

    return {
        "horizon_days": horizon,
        "folds_completed": completed,
        "observations": len(values),
        "series_period": [str(series.index.min().date()), str(series.index.max().date())],
        "results": summary,
        "best_model": best,
        "beats_naive": bool(naive and best != "naive" and summary[0]["mape_pct"] < naive["mape_pct"]),
        "lstm_evaluated": "lstm" in collected and bool(collected["lstm"]),
        "torch_available": torch_available(),
        "sarima_convergence": {
            "converged_fits": int(sum(SARIMA_CONVERGENCE)),
            "total_fits": len(SARIMA_CONVERGENCE),
        },
    }


def forecast_ahead(horizon: int = 14, model: str = "sarima") -> list[dict]:
    """Produce the forward forecast with the chosen model."""
    series = load_series()
    values = series.to_numpy(dtype=float)

    function = {
        "naive": forecast_naive,
        "drift": forecast_drift,
        "seasonal_naive": forecast_seasonal_naive,
        "sarima": forecast_sarima,
        "lstm": forecast_lstm,
    }[model]

    predicted = function(values, horizon)
    last_date = series.index.max()
    return [
        {
            "forecast_date": str((last_date + pd.Timedelta(days=index + 1)).date()),
            "predicted_value": round(float(value), 2),
        }
        for index, value in enumerate(predicted)
    ]


def run(horizon: int = 14, folds: int = 6, save: bool = True) -> dict:
    report = backtest(horizon=horizon, folds=folds)
    report["forward_forecast"] = forecast_ahead(horizon, report["best_model"] or "sarima")

    if save:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "forecasting_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    return report


if __name__ == "__main__":
    print("Rolling-origin backtest ...")
    outcome = run()

    print(f"\n{outcome['folds_completed']} folds, {outcome['horizon_days']}-day horizon, "
          f"{outcome['observations']} observations")
    print(f"{'model':<16}{'MAE':>14}{'RMSE':>14}{'MAPE %':>10}{'direction %':>14}")
    for row in outcome["results"]:
        direction = ("n/a" if row["directional_accuracy_pct"] is None
                     else f"{row['directional_accuracy_pct']:.1f}")
        print(f"{row['model']:<16}{row['mae']:>14,.0f}{row['rmse']:>14,.0f}"
              f"{row['mape_pct']:>10.3f}{direction:>14}")

    convergence = outcome["sarima_convergence"]
    print(f"SARIMA converged on {convergence['converged_fits']}/"
          f"{convergence['total_fits']} fits.")
    print(f"\nBest: {outcome['best_model']}  |  beats naive: {outcome['beats_naive']}")
    if not outcome["lstm_evaluated"]:
        print("LSTM not evaluated (PyTorch unavailable).")
