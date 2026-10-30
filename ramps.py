"""Ramp-up schedules used by the semi-supervised training utilities."""

import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Compute the exponential sigmoid ramp-up from the Mean Teacher method."""
    if rampup_length == 0:
        return 1.0

    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))
