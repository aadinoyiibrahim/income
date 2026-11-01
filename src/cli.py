"""Command-line entry point: ``income <command>`` (or ``python -m src.cli``).

income train     # evaluate on hold-out users, refit on all data, save model bundle
income predict   # forecast next month for every user in a CSV
income report    # figures from the saved bundle + hold-out results
income eda       # exploratory figures and statistics
income serve     # run the API
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .config import get_paths

log = logging.getLogger("income")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_train(args: argparse.Namespace) -> None:
    from .bundle import save_bundle
    from .preprocessing import load_lookups, read_transactions
    from .train_model import TrainConfig, run_training

    paths = get_paths()
    tx = read_transactions(args.data or paths.training_csv)
    cfg = TrainConfig(n_clusters=args.clusters, n_trials=args.trials, holdout_frac=args.holdout_frac, seed=args.seed)
    bundle, holdout = run_training(tx, load_lookups(paths.data_dir), cfg)
    save_bundle(bundle, args.model or paths.model_path)

    paths.results_dir.mkdir(parents=True, exist_ok=True)
    holdout.to_csv(paths.results_dir / "holdout_predictions.csv", index=False)
    (paths.results_dir / "metrics.json").write_text(json.dumps(bundle.metrics, indent=2))
    log.info("Wrote %s and metrics.json", paths.results_dir / "holdout_predictions.csv")


def cmd_predict(args: argparse.Namespace) -> None:
    from .bundle import load_bundle
    from .predict import predict_next_month
    from .preprocessing import read_transactions

    paths = get_paths()
    bundle = load_bundle(args.model or paths.model_path)
    preds = predict_next_month(bundle, read_transactions(args.input))
    out = Path(args.output or paths.results_dir / f"predictions_{preds['forecast_month'].iloc[0]}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.assign(
        predicted_income_breakdown=preds["predicted_income_breakdown"].map(json.dumps),
        predicted_expenses_breakdown=preds["predicted_expenses_breakdown"].map(json.dumps),
    ).to_csv(out, index=False)
    log.info("Forecast %s for %d users -> %s", preds["forecast_month"].iloc[0], len(preds), out)


def _headless_matplotlib():
    import matplotlib

    matplotlib.use("Agg")  # no display in CLI / CI / Docker
    import matplotlib.pyplot as plt

    return plt


def cmd_report(args: argparse.Namespace) -> None:
    plt = _headless_matplotlib()
    import pandas as pd

    from . import plotter
    from .bundle import load_bundle
    from .cluster import cluster_users
    from .preprocessing import build_feature_frame, read_transactions

    paths = get_paths()
    outdir_ = str(paths.img_dir) + "/"
    bundle = load_bundle(args.model or paths.model_path)
    holdout = pd.read_csv(paths.results_dir / "holdout_predictions.csv")
    feats = cluster_users(
        build_feature_frame(read_transactions(args.data or paths.training_csv), bundle.lookups), bundle.clusterer
    )
    for name, make in (
        ("evaluation_overview", lambda: plotter.plot_evaluation(holdout, outdir_)),
        ("feature_importance", lambda: plotter.plot_feature_importance(bundle, outdir_)),
        ("clusters", lambda: plotter.plot_clusters(feats, outdir_)),
        ("shap_summary", lambda: plotter.plot_shap_summary(bundle, feats, outdir_)),
    ):
        plt.close(make())
        log.info("Saved %s%s.png", outdir_, name)


def cmd_eda(args: argparse.Namespace) -> None:
    plt = _headless_matplotlib()
    from .eda import perform_eda
    from .preprocessing import load_lookups, read_transactions

    paths = get_paths()
    summary, fig = perform_eda(
        read_transactions(args.data or paths.training_csv),
        load_lookups(paths.data_dir),
        str(paths.img_dir / "eda") + "/",
    )
    plt.close(fig)
    print(json.dumps(summary, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("src.api:app", host=args.host, port=args.port, reload=args.reload)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="income", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="evaluate on hold-out users, then fit final model")
    t.add_argument("--data", help="training CSV (default: data/2016-09-19_79351_training.csv)")
    t.add_argument("--model", help="output bundle path (default: models/model_bundle.joblib)")
    t.add_argument("--trials", type=int, default=25, help="Optuna trials per model; 0 = default params (fast)")
    t.add_argument("--clusters", type=int, default=5)
    t.add_argument("--holdout-frac", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="forecast next month for users in a CSV")
    pr.add_argument("--input", required=True, help="transactions CSV (≥1 complete month per user)")
    pr.add_argument("--output", help="output CSV (default: results/predictions_<month>.csv)")
    pr.add_argument("--model")
    pr.set_defaults(func=cmd_predict)

    r = sub.add_parser("report", help="evaluation / importance / cluster figures into img/")
    r.add_argument("--data")
    r.add_argument("--model")
    r.set_defaults(func=cmd_report)

    e = sub.add_parser("eda", help="EDA figure into img/eda/ and summary stats")
    e.add_argument("--data")
    e.set_defaults(func=cmd_eda)

    s = sub.add_parser("serve", help="run the FastAPI app with uvicorn")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
