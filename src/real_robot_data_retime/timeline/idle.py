"""Shared review idle rules reused from the drawer-task exporter."""

import numpy as np

from ..retime import boolean_ranges


def clock_idle_mask(clock):
    """Mask duplicate source-clock samples, keeping one supervised boundary.

    Leading/interior holds retain the departure sample. Terminal holds retain
    the arrival sample. A constant clock has no execution and is entirely idle.
    Equality is exact: small positive speed-ramp steps remain supervised.
    """
    clock = np.asarray(clock, dtype=float)
    if clock.ndim != 1 or not len(clock) or not np.isfinite(clock).all():
        raise ValueError("Expected a nonempty finite source clock")
    if np.any(np.diff(clock) < 0):
        raise ValueError("Source clock must be monotone")
    idle = np.zeros(len(clock), dtype=bool)
    if len(clock) > 1 and np.all(clock == clock[0]):
        idle[:] = True
        return idle
    for span in boolean_ranges(np.diff(clock) == 0):
        if span.end == len(clock) - 1:
            idle[span.start + 1 : span.end + 1] = True
        else:
            idle[span.start : span.end] = True
    return idle


def terminal_rest_limit(state, action, after_frame, fps):
    """Bound the entire final pose range; never bridge later return movement.

    Same pose tolerances as timeline.holds: 0.3deg joints and 0.5mm gripper.
    Both measured and commanded poses must stay bounded until episode end.
    """
    state, action = np.asarray(state, float), np.asarray(action, float)
    if (
        state.ndim != 2
        or state.shape[1] != 7
        or state.shape != action.shape
        or not len(state)
        or not np.isfinite([state, action]).all()
    ):
        raise ValueError("Expected matching finite N x 7 state/action arrays")
    if not np.isfinite(fps) or fps <= 0 or not 0 <= after_frame < len(state):
        raise ValueError("Invalid FPS or completion frame")
    values = np.c_[state, action]
    limit = np.tile([0.3] * 6 + [0.5], 2)
    low, high = values[-1].copy(), values[-1].copy()
    start = len(values) - 1
    for index in range(len(values) - 2, after_frame - 1, -1):
        lo, hi = np.minimum(low, values[index]), np.maximum(high, values[index])
        if np.any(hi - lo > limit):
            break
        low, high, start = lo, hi, index
    keep = int(np.floor(1.5 * fps))
    cutoff = min(len(values), start + keep)
    return {
        "start": start,
        "mask_start": cutoff,
        "end": len(values),
        "max_supervised_frames": keep,
    }


def screw_idle_masks(left, right, state, action, stages, fps):
    """Exclude onset delays; preserve dependency waits and protected actions."""
    clocks = [np.asarray(left), np.asarray(right)]
    state, action = np.asarray(state), np.asarray(action)
    n = len(clocks[0])
    if clocks[1].shape != (n,) or state.shape != (n, 14) or action.shape != state.shape:
        raise ValueError("idle motion arrays do not match output frames")
    masks = [clock_idle_mask(clock) for clock in clocks]
    required = [np.zeros(n, bool), np.zeros(n, bool)]
    protected = [np.zeros(n, bool), np.zeros(n, bool)]
    for stage in stages:
        a, b = stage["output_start"], stage["output_end"] + 1
        if stage["kind"] == "coupled":
            for side in range(2):
                protected[side][a:b] = True
        if stage["kind"] == "independent":
            for side, tr in enumerate(stage["transitions"]):
                onset = stage["pickup_start_output_frames"][side]
                masks[side][onset:b] = False
                if tr["hold_frames"]:
                    required[side][
                        tr["arrival_output_frame"] : tr["restart_start_output_frame"]
                    ] = True
        if "mandatory_previous_right_retreat" in stage:
            r = stage["mandatory_previous_right_retreat"]
            if r["previous_cycle"] > 0:
                protected[1][r["output_start"] : r["output_end"] + 1] = True
    last = next(s for s in reversed(stages) if s["kind"] == "coupled")
    completed = [last["output_end"], last["right_retreat"]["output_end"]]
    record = {}
    for side, arm in enumerate(["left", "right"]):
        sl = slice(side * 7, side * 7 + 7)
        rest = terminal_rest_limit(state[:, sl], action[:, sl], completed[side], fps)
        masks[side][rest["mask_start"] :] = True
        masks[side][protected[side]] = False
        rest["supervised_frames"] = int(np.count_nonzero(~masks[side][rest["start"] :]))
        record[arm + "_terminal_rest"] = rest
        record[arm] = [[r.start, r.end] for r in boolean_ranges(masks[side])]
        record[arm + "_required_wait"] = [
            [r.start, r.end] for r in boolean_ranges(required[side])
        ]
    return record


def mask_ranges(ranges, n):
    """Expand ordered, half-open annotation intervals without changing labels."""
    result = np.zeros(n, bool)
    last = 0
    for a, b in ranges:
        if not 0 <= last <= a < b <= n:
            raise ValueError("invalid idle intervals")
        result[a:b] = True
        last = b
    return result


def arm_active_mask(labels, n):
    """Float32 loss weights in [left, right] order, preserving idle annotations."""
    return np.column_stack(
        [~mask_ranges(labels[arm], n) for arm in ("left", "right")]
    ).astype(np.float32)
