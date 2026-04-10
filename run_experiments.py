#!/usr/bin/env python3
"""Run TLOB/MLPLOB experiments with three modes:

single         - one run for a single --horizon value
all-horizons   - four sequential runs, one per horizon (default)
multi-horizons - one joint run training all horizons simultaneously

Examples:
  python run_experiments.py --mode multi-horizons --model tlob --dataset battery
  python run_experiments.py --dataset battery --sampling-time 5s --rebuild-data
  python run_experiments.py --dataset battery --sampling-time 10s --dates 2021-01-11 2021-01-15
  python run_experiments.py --dataset battery --mode all-horizons --sampling-time 5s --dry-run
"""

import argparse
import os
import subprocess
import sys

MODEL = "tlob"
DATASET = "fi_2010"
HORIZONS = [10, 20, 50, 100]
SEED = 1
MAX_EPOCHS = 50
IS_WANDB = True


def profile_overrides(args):
    """Return Hydra overrides for a named experiment profile."""
    profile = args.profile.lower()
    is_original_model = str(args.model).lower().endswith("_original")

    if profile == "auto":
        # Backward-compatible behavior with safe baseline defaults.
        if is_original_model:
            return ["experiment.use_diff_features=False"]
        return []

    if profile == "strict_original":
        return [
            "experiment.use_diff_features=False",
            "experiment.use_class_weights=False",
            "experiment.label_mode=percent_change",
            "experiment.loss_type=cross_entropy",
            "experiment.optimizer=Adam",
            "experiment.max_epochs=10",
            "experiment.use_torch_compile=False",
            "experiment.use_fast_attention=False",
        ]

    if profile == "ablation":
        # Intentionally minimal profile, preserves CLI-provided settings.
        return []

    raise ValueError(f"Unknown profile: {args.profile}")


def run_command(command, dry_run=False):
    print(f"\nExecuting: {' '.join(command)}")
    if not dry_run:
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while running command: {e}")
            sys.exit(1)
    else:
        print("[DRY-RUN] Command would be executed here.")


def base_command(args, is_preprocessed="True"):
    """Build the common part of every main.py invocation."""
    command = [
        sys.executable,
        "main.py",
        f"+model={args.model}",
        f"+dataset={args.dataset}",
        "hydra.job.chdir=False",
        f"experiment.seed={SEED}",
        f"experiment.max_epochs={args.epochs}",
        f"experiment.is_data_preprocessed={is_preprocessed}",
        f"experiment.is_wandb={not args.no_wandb and IS_WANDB}",
    ]
    command += profile_overrides(args)
    if args.loss_type is not None:
        command.append(f"experiment.loss_type={args.loss_type}")
    if args.encoder_checkpoint:
        # Set via environment variable to avoid Hydra parsing issues with '=' in paths
        os.environ["TRADELOB_ENCODER_CHECKPOINT"] = args.encoder_checkpoint
    return command


def battery_extras(args):
    extras = [
        "dataset.training_stocks=[battery_markets]",
        "dataset.testing_stocks=[battery_markets]",
    ]
    if args.sampling_time is not None:
        extras.append(f"dataset.sampling_time={args.sampling_time}")
    if args.dates is not None:
        extras.append(f"dataset.dates=[{args.dates[0]},{args.dates[1]}]")
    return extras


def main():
    parser = argparse.ArgumentParser(
        description="Run TLOB/MLPLOB experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["single", "all-horizons", "multi-horizons"],
        default="all-horizons",
        help="Experiment mode (default: all-horizons).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Horizon for 'single' mode (default: 10).",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        default=HORIZONS,
        help=f"Horizons for 'all-horizons' mode (default: {HORIZONS}).",
    )
    parser.add_argument(
        "--model", type=str, default=MODEL, help=f"Model to use (default: {MODEL})"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help=f"Dataset to use (default: {DATASET})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"Max epochs per run (default: {MAX_EPOCHS})",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Force data preprocessing on the first run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing them."
    )
    parser.add_argument(
        "--sampling-time",
        type=str,
        default=None,
        help="Battery sampling interval (e.g. '5s', '10s')",
    )
    parser.add_argument(
        "--dates",
        type=str,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Date range (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging (useful for quick validation runs).",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["auto", "strict_original", "ablation"],
        default="auto",
        help=(
            "Preset policy for experiment overrides. "
            "auto keeps existing behavior; strict_original enforces paper-like baseline defaults."
        ),
    )
    parser.add_argument(
        "--loss-type",
        type=str,
        default=None,
        help="Override loss type (cross_entropy, focal, dfl_proxy, dfl_trading).",
    )
    parser.add_argument(
        "--encoder-checkpoint",
        type=str,
        default="",
        help="Path to pre-trained TLOB checkpoint for TradeLOB encoder loading.",
    )

    args = parser.parse_args()

    is_battery = args.dataset.lower() == "battery"
    if args.mode == "multi-horizons" and str(args.model).lower().endswith("_original"):
        raise SystemExit(
            "multi-horizons mode is not supported for *_original baseline models. Use --mode single or all-horizons."
        )

    # ----------------------------------------------------------------- #
    # single: one run for a specified horizon
    # ----------------------------------------------------------------- #
    if args.mode == "single":
        is_preprocessed = "False" if args.rebuild_data else "True"
        cmd = base_command(args, is_preprocessed)
        cmd += [f"experiment.horizon={args.horizon}"]
        if is_battery:
            cmd += battery_extras(args)
        run_command(cmd, dry_run=args.dry_run)

    # ----------------------------------------------------------------- #
    # all-horizons: four sequential single-horizon runs
    # ----------------------------------------------------------------- #
    elif args.mode == "all-horizons":
        horizons = args.horizons if isinstance(args.horizons, list) else [args.horizons]
        for i, h in enumerate(horizons):
            is_preprocessed = "False" if args.rebuild_data and i == 0 else "True"
            cmd = base_command(args, is_preprocessed)
            cmd += [f"experiment.horizon={h}"]
            if is_battery:
                cmd += battery_extras(args)
            run_command(cmd, dry_run=args.dry_run)

    # ----------------------------------------------------------------- #
    # multi-horizons: one joint run training all four horizons together
    # ----------------------------------------------------------------- #
    elif args.mode == "multi-horizons":
        is_preprocessed = "False" if args.rebuild_data else "True"
        cmd = base_command(args, is_preprocessed)
        cmd += ["experiment.multi_horizon=True"]
        if is_battery:
            cmd += battery_extras(args)
        run_command(cmd, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
