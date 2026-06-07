import json
import os
from datetime import datetime


def setup_run_dir(experiment_name="hd_experiment"):
    """Create a unique directory for the current training run."""
    # Format: "YY-MM-DD_HHMM_experiment_name"
    timestamp = datetime.now().strftime("%y-%m-%d_%H%M")
    run_name = f"{timestamp}_{experiment_name}"
    run_dir = os.path.join("results", run_name)

    # Create directories
    os.makedirs(run_dir, exist_ok=True)

    print(f"Saving all results to: {run_dir}")
    return run_dir


def save_config(run_dir, config):
    """Save hyperparameters dictionary as a formatted JSON file."""
    config_path = os.path.join(run_dir, "config.json")

    with open(config_path, "w", encoding="utf-8") as f:
        # indent=4 makes the JSON file human-readable
        json.dump(config, f, indent=4, sort_keys=True)

    print(f"Configuration saved to: {config_path}")
