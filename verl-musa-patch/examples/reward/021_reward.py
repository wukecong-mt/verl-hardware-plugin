"""VERL reward adapter for the 021/zero2one math task."""

import importlib.util
from functools import lru_cache
from pathlib import Path


DEFAULT_SOURCE = str(Path(__file__).with_name("zero2one_reward.py"))


@lru_cache(maxsize=None)
def _load_source(path: str):
    source = Path(path)
    if not source.is_absolute():
        source = Path(__file__).with_name(source.name)
    path = str(source)
    spec = importlib.util.spec_from_file_location("slime_zero2one_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load 021 reward implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    source_path=DEFAULT_SOURCE,
    **kwargs,
):
    """Return the binary accuracy expected by Slime's ``rm_type=zero2one``."""
    del data_source, extra_info, kwargs
    is_correct, _ = _load_source(source_path).is_correct_int(
        solution_str.strip(), ground_truth
    )
    score = float(is_correct)
    return {"score": score, "accuracy_score": score}
