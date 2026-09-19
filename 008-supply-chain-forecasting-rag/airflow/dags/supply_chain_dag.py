"""Daily supply chain pipeline.

Shape of the DAG, and why:

    ingest -> features -> [train_model, planning_views] -> validate -> publish

`train_model` and `planning_views` both depend on features and not on each other,
so they run in parallel. `validate` is a real gate, not a logging step: if the
retrained model is worse than the planning rule it is meant to replace, the task
fails and yesterday's model keeps serving. A pipeline that publishes whatever it
produced is a pipeline that will eventually publish something broken at 03:00 and
tell nobody.
"""
from __future__ import annotations

import json
import pendulum
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException

DEFAULT_ARGS = {
    "owner": "supply-chain-platform",
    "retries": 2,
    "retry_delay": pendulum.duration(minutes=5),
    "email_on_failure": True,
}

# Retraining is refused if the new model is not at least this much better than
# the existing planning rule, measured on the held-out period.
MIN_IMPROVEMENT_OVER_RULE = 0.15
# And refused if it has regressed badly against the model already in production.
MAX_REGRESSION_VS_PRODUCTION = 0.10


@dag(
    dag_id="supply_chain_daily",
    description="Shipment features, delivery-time model, and store planning views.",
    schedule="0 3 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=DEFAULT_ARGS,
    tags=["supply-chain", "forecasting", "spark"],
)
def supply_chain_daily():

    @task
    def ingest() -> dict:
        """In production this lands yesterday's shipments from the source system
        onto object storage. Here it regenerates the synthetic dataset."""
        from data.generate import generate

        return generate()

    @task
    def build_features() -> dict:
        """Spark feature engineering. Falls back to pandas where no JVM exists,
        which is what makes this DAG runnable on a laptop."""
        from pipeline.features import build_features, spark_available, write_curated

        frame = build_features()
        path = write_curated(frame)
        return {
            "engine": "spark" if spark_available() else "pandas",
            "rows": int(len(frame)),
            "usable_rows": int((~frame["in_warmup"]).sum()),
            "path": str(path),
        }

    @task
    def train_model() -> dict:
        from pipeline.train_delivery_model import train

        # save=False: nothing is published until validate has passed.
        return train(save=False)

    @task
    def planning_views() -> dict:
        from pipeline.planning_views import build_all

        return build_all()

    @task
    def validate(metrics: dict) -> dict:
        """Gate. Refuses to publish a model that is not good enough."""
        model_mae = metrics["model"]["mae"]
        rule_mae = metrics["baseline_planning_rule"]["mae"]
        improvement = 1 - model_mae / rule_mae

        if improvement < MIN_IMPROVEMENT_OVER_RULE:
            raise AirflowFailException(
                f"Model MAE {model_mae:.3f} is only {improvement:.1%} better than the "
                f"planning rule ({rule_mae:.3f}); minimum is "
                f"{MIN_IMPROVEMENT_OVER_RULE:.0%}. Not publishing."
            )

        from pathlib import Path

        production = Path(__file__).resolve().parents[2] / "models" / "delivery_time_metrics.json"
        if production.exists():
            previous = json.loads(production.read_text(encoding="utf-8"))["model"]["mae"]
            if model_mae > previous * (1 + MAX_REGRESSION_VS_PRODUCTION):
                raise AirflowFailException(
                    f"Model MAE {model_mae:.3f} has regressed more than "
                    f"{MAX_REGRESSION_VS_PRODUCTION:.0%} against production "
                    f"({previous:.3f}). Not publishing."
                )

        return {"improvement_over_rule_pct": round(improvement * 100, 2), **metrics["model"]}

    @task
    def publish(validation: dict) -> dict:
        """Persist the model and its metrics, then let the API pick them up.

        Training runs twice as a result: once to evaluate, once to publish. That
        is a few CPU-minutes to guarantee the artifact on disk is exactly the one
        that was validated, which is worth more than the minutes.
        """
        from pipeline.train_delivery_model import train

        published = train(save=True)
        return {
            "published": True,
            "mae": published["model"]["mae"],
            "improvement_over_rule_pct": validation["improvement_over_rule_pct"],
        }

    raw = ingest()
    features = build_features()
    raw >> features

    metrics = train_model()
    views = planning_views()
    features >> [metrics, views]

    publish(validate(metrics))


supply_chain_daily()
