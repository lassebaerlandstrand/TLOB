#!/usr/bin/env python3
"""Run TLOB/MLPLOB experiments with three modes:

  single       - one run for a single --horizon value
  all-horizons - four sequential runs, one per horizon (default)
  multi-horizons- one joint run training all horizons simultaneously
"""
import subprocess
import sys
import argparse

MODEL = "tlob"
DATASET = "fi_2010"
HORIZONS = [10, 20, 50, 100]
SEED = 1
MAX_EPOCHS = 20
IS_WANDB = "True"
BATTERY_START_DATE = "2021-01-11"
BATTERY_END_DATE = "2021-01-21"


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
    return [
        sys.executable, "main.py",
        f"+model={args.model}",
        f"+dataset={args.dataset}",
        "hydra.job.chdir=False",
        f"experiment.seed={SEED}",
        f"experiment.max_epochs={args.epochs}",
        f"experiment.is_data_preprocessed={is_preprocessed}",
        f"experiment.is_wandb={IS_WANDB}",
    ]


def battery_extras(args):
    return [
        f"dataset.dates=[{args.start_date},{args.end_date}]",
        "dataset.training_stocks=[battery_markets]",
        "dataset.testing_stocks=[battery_markets]",
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Run TLOB/MLPLOB experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["single", "all-horizons", "multi-horizons"],
                        default="all-horizons",
                        help="Experiment mode (default: all-horizons).")
    parser.add_argument("--horizon", type=int, default=10,
                        help="Horizon for 'single' mode (default: 10).")
    parser.add_argument("--horizons", type=int, nargs="*", default=HORIZONS,
                        help=f"Horizons for 'all-horizons' mode (default: {HORIZONS}).")
    parser.add_argument("--model", type=str, default=MODEL,
                        help=f"Model to use (default: {MODEL})")
    parser.add_argument("--dataset", type=str, default=DATASET,
                        help=f"Dataset to use (default: {DATASET})")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS,
                        help=f"Max epochs per run (default: {MAX_EPOCHS})")
    parser.add_argument("--start-date", type=str, default=BATTERY_START_DATE)
    parser.add_argument("--end-date", type=str, default=BATTERY_END_DATE)
    parser.add_argument("--rebuild-data", action="store_true",
                        help="Force data preprocessing on the first run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them.")

    args = parser.parse_args()

    is_battery = args.dataset.lower() == "battery"

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
