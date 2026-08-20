from pathlib import Path
from datetime import datetime, timezone

import os
import pickle
import numpy as np


CHECKPOINT_VERSION = 1


def save_bootstrap_checkpoint(
    checkpoint_path,
    *,
    completed,
    attempts,
    n_bootstrap,
    bootstrap_results,
    bootstrap_ig_stat_results,
    failure_log,
    rng,
    metadata=None,
):
    """
    Atomically save all completed bootstrap results.

    bootstrap_mos:
        Optional list containing one MOS vector per accepted replicate.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "version": CHECKPOINT_VERSION,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "attempts": attempts,
        "n_bootstrap": n_bootstrap,
        "bootstrap_results": bootstrap_results,
        "bootstrap_ig_stat_results": bootstrap_ig_stat_results,
        "failure_log": failure_log,
        "rng_state": rng.bit_generator.state,
        "metadata": metadata or {},
    }

    temporary_path = checkpoint_path.with_suffix(
        checkpoint_path.suffix + ".tmp"
    )

    with temporary_path.open("wb") as file:
        pickle.dump(
            state,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
        file.flush()
        os.fsync(file.fileno())

    # Atomic replacement: the previous checkpoint remains intact if writing fails.
    os.replace(temporary_path, checkpoint_path)


def load_bootstrap_checkpoint(checkpoint_path, rng):
    checkpoint_path = Path(checkpoint_path)

    with checkpoint_path.open("rb") as file:
        state = pickle.load(file)

    if state["version"] != CHECKPOINT_VERSION:
        raise RuntimeError(
            f"Unsupported checkpoint version: {state['version']}"
        )

    rng.bit_generator.state = state["rng_state"]

    validate_bootstrap_checkpoint(state)

    return state

def validate_bootstrap_checkpoint(state):
    completed = state["completed"]

    for metric, statistics in state["bootstrap_results"].items():
        for statistic in ("plcc", "srocc", "or"):
            values = statistics[statistic]

            if len(values) != completed:
                raise RuntimeError(
                    f"{metric}/{statistic} contains {len(values)} "
                    f"values, expected {completed}."
                )

            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"{metric}/{statistic} contains non-finite values."
                )

    for property_name, statistics in (
        state["bootstrap_ig_stat_results"].items()
    ):
        for statistic in ("plcc", "srocc"):
            values = statistics[statistic]

            if len(values) != completed:
                raise RuntimeError(
                    f"{property_name}/{statistic} contains "
                    f"{len(values)} values, expected {completed}."
                )

            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"{property_name}/{statistic} contains "
                    "non-finite values."
                )

    bootstrap_mos = state.get("bootstrap_mos")

    if bootstrap_mos is not None:
        if len(bootstrap_mos) != completed:
            raise RuntimeError(
                "bootstrap_mos has an inconsistent length."
            )

