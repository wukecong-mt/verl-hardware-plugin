"""Local 021/zero2one answer checker used by the VERL reward adapter."""

import re


def is_correct_int(solution_str, ground_truth):
    """Match the last ``\\boxed{...}`` answer as an integer/float."""
    matches = re.findall(r"\\boxed\{(.*?)\}", solution_str, re.DOTALL)
    if not matches:
        return False, "[INVALID]"

    pred_str = matches[-1]
    try:
        if float(pred_str) == float(ground_truth):
            return True, pred_str
    except (TypeError, ValueError):
        pass
    return False, pred_str
