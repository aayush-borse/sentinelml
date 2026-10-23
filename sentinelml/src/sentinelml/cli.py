import argparse
import json
from pathlib import Path
import pandas as pd
from sentinelml.artifacts import active_paths
from sentinelml.config import DATA_DIR, FEATURE_NAMES
from sentinelml.data import generate_dataset
from sentinelml.monitoring import monitor_frame
from sentinelml.training import train_model
from sentinelml.validation import validate_frame

def main():
    parser = argparse.ArgumentParser(description="SentinelML CLI")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("train", help="Train a candidate and promote it if quality gates pass.")
    subs.add_parser("validate", help="Validate generated training data.")
    subs.add_parser("model-info", help="Print metadata for the active model.")
    monitor = subs.add_parser("monitor", help="Check a CSV batch for feature drift.")
    monitor.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "train":
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        frame = generate_dataset()
        frame.to_csv(DATA_DIR / "training_data.csv", index=False)
        print(json.dumps(train_model(frame), indent=2))
    elif args.command == "validate":
        report = validate_frame(generate_dataset(), require_target=True)
        print(json.dumps(report.to_dict(), indent=2))
        if not report.valid:
            raise SystemExit(1)
    elif args.command == "model-info":
        paths = active_paths()
        print(paths["metadata"].read_text(encoding="utf-8"))
    elif args.command == "monitor":
        if not args.input.exists():
            raise SystemExit(f"Input file does not exist: {args.input}")
        paths = active_paths()
        frame = pd.read_csv(args.input)
        missing = sorted(set(FEATURE_NAMES) - set(frame.columns))
        if missing:
            raise SystemExit(f"CSV missing features: {', '.join(missing)}")
        print(json.dumps(monitor_frame(frame[FEATURE_NAMES], paths["baseline"]), indent=2))

if __name__ == "__main__":
    main()
