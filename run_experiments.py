#!/usr/bin/env python3
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

def main():
    parser = argparse.ArgumentParser(description="Run TLOB experiments across multiple horizons.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--model", type=str, default=MODEL, help=f"Model to use (default: {MODEL})")
    parser.add_argument("--dataset", type=str, default=DATASET, help=f"Dataset to use (default: {DATASET})")
    parser.add_argument("--horizons", type=int, nargs="*", default=HORIZONS, help=f"Horizons to test (default: {HORIZONS})")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help=f"Max epochs (default: {MAX_EPOCHS})")
    parser.add_argument("--start-date", type=str, default=BATTERY_START_DATE, help=f"Battery start date (default: {BATTERY_START_DATE})")
    parser.add_argument("--end-date", type=str, default=BATTERY_END_DATE, help=f"Battery end date (default: {BATTERY_END_DATE})")
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Rebuild preprocessed data on the first horizon run, then reuse for the remaining runs.",
    )
    
    args = parser.parse_args()

    horizons = args.horizons if isinstance(args.horizons, list) else [args.horizons]

    for i, h in enumerate(horizons):
        is_preprocessed = "False" if args.rebuild_data and i == 0 else "True"

        command = [
            sys.executable, "main.py",
            f"+model={args.model}",
            f"+dataset={args.dataset}",
            "hydra.job.chdir=False",
            f"experiment.horizon={h}",
            f"experiment.seed={SEED}",
            f"experiment.max_epochs={args.epochs}",
            f"experiment.is_data_preprocessed={is_preprocessed}",
            f"experiment.is_wandb={IS_WANDB}"
        ]

        if args.dataset.lower() == "battery":
            command.extend([
                f"dataset.dates=[{args.start_date},{args.end_date}]",
                "dataset.training_stocks=[battery_markets]",
                "dataset.testing_stocks=[battery_markets]",
            ])
        
        run_command(command, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
