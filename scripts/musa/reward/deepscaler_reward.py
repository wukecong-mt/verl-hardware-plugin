"""VERL adapter for Slime's DeepScaler/DeepScalar math reward."""

import importlib
import os
import sys
import threading
import types
from functools import cache
from pathlib import Path

DEFAULT_SOURCE_DIR = str(Path(__file__).resolve().parent)
_LOAD_LOCK = threading.Lock()


@cache
def _load_reward(source_dir: str):
    """Load the enhanced DeepScaler math reward shipped with this plugin."""
    # RewardLoop evaluates samples from a thread pool. functools.cache may
    # invoke the wrapped function more than once during the first concurrent
    # cache miss, so serialize module creation. Otherwise one thread can read
    # the shared sys.modules entry while another thread has registered but not
    # yet executed deepscaler.py, producing a module without its reward API.
    with _LOAD_LOCK:
        package_name = "verl_local_deepscaler_math_v2"
        package_dir = os.path.join(source_dir, "deepscaler_math_v2")
        package = types.ModuleType(package_name)
        package.__path__ = [package_dir]
        sys.modules[package_name] = package
        return importlib.import_module(f"{package_name}.math_reward")


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    source_dir=DEFAULT_SOURCE_DIR,
    **kwargs,
):
    """Return the rule-based DeepScaler correctness score."""
    del data_source, extra_info, kwargs
    score = float(
        _load_reward(source_dir).deepscaler_reward_fn(
            solution_str.strip(), ground_truth
        )
    )
    return {"score": score, "accuracy_score": score}
