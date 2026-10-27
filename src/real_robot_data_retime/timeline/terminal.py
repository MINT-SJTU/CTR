"""Measure recorded final rest and extend it with explicitly synthetic holds."""

import math

import numpy as np

from ..trim import analyze_episode


def terminal_hold_plan(state, action, fps, *, seconds=1.5):
    """Require both commanded and measured arms to be static at episode end.

    Reuse trimming's conservative 0.1 degree/s and 0.1 mm/s thresholds.
    A single final sample cannot establish rest. Synthetic frames repeat the
    last measured state and recorded command, without claiming physical settling.
    """
    state, action = np.asarray(state), np.asarray(action)
    if state.shape != action.shape:
        raise ValueError("state/action shapes differ")
    if not np.isfinite(seconds) or seconds <= 0:
        raise ValueError("terminal hold duration must be positive")
    evidence = [analyze_episode(x, fps, tail_seconds=seconds) for x in (state, action)]
    recorded = min(
        e[side]["tail_frames"] for e in evidence for side in ("left", "right")
    )
    if len(state) < 2:
        recorded = 0
    target = math.ceil(seconds * fps)
    return {
        "recorded_static_frames": recorded,
        "recorded_static_start": len(state) - recorded,
        "target_static_frames": target,
        "added_frames": max(0, target - recorded),
        "fps": fps,
        "seconds": seconds,
        "joint_velocity_threshold_deg_s": 0.1,
        "gripper_velocity_threshold_mm_s": 0.1,
    }
