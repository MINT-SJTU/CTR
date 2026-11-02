"""Water timing from frozen commands; one scheduler for manual and uniform samples."""

from dataclasses import asdict, dataclass
import math

import numpy as np
from scipy.ndimage import median_filter

from ..retime import (
    ArmSegment,
    EpisodeSegments,
    boolean_ranges,
    build_uniform_schedule_plan,
)
from ..trim import analyze_episode

ARMS = ("left", "right")


@dataclass(frozen=True)
class Timing:
    first_arm: str
    delay_ratio: float

    def __post_init__(self):
        if self.first_arm not in ARMS or not 0 <= self.delay_ratio <= 1:
            raise ValueError(
                "first_arm must be left/right and delay_ratio must be in [0, 1]"
            )


def detect_phases(action, state, fps=30, *, motion_threshold=3.0):
    """Inclusive event frames; aperture >20 mm starts an opening command.

    Motion thresholds are degrees/s and mm/s, as in trim.analyze_episode.
    Durations come from frozen actions, never a dataset average.
    """
    action, state = np.asarray(action), np.asarray(state)
    if action.ndim != 2 or action.shape[1] != 14 or state.shape != action.shape:
        raise ValueError("action and state must both have shape (N, 14)")
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise ValueError("nonfinite action/state")
    n = len(action)
    trim = analyze_episode(
        action, fps, motion_threshold, motion_threshold, tail_seconds=0
    )
    cycles = []
    for column in (6, 13):
        opened = median_filter(action[:, column], size=3) > 20
        events = [
            (r.start, r.end)
            for r in boolean_ranges(opened)
            if r.length >= max(2, math.ceil(0.2 * fps))
        ]
        if len(events) < 3:
            raise ValueError("missing bottle/tray gripper opening cycles")
        cycles.append(events)
    # A paired opening can be a regrasp before carrying the tray. Select the
    # release following the longest continuous bilateral closed interval;
    # brief regrips and later empty-gripper cycles must not cut the carry.
    paired = [
        (left, right, max(cycles[0][li - 1][1], cycles[1][ri - 1][1]))
        for li, left in enumerate(cycles[0][2:], 2)
        for ri, right in enumerate(cycles[1][2:], 2)
        if abs(left[0] - right[0]) <= math.ceil(2 * fps / 3)
    ]
    if not paired:
        raise ValueError("no paired tray release")
    release = max(
        paired, key=lambda pair: min(pair[0][0], pair[1][0]) - pair[2]
    )[:2]
    ends = [c[1][0] + math.ceil(0.6 * fps) for c in cycles]
    shared_start = max(ends)
    shared_end = max(x[0] for x in release) + math.ceil(0.2 * fps)
    starts = [trim[arm]["head_frames"] for arm in ARMS]
    returns = [n - trim[arm]["tail_frames"] - 1 for arm in ARMS]
    if not all(
        0 <= starts[i] < ends[i] <= shared_start < shared_end < returns[i] < n
        for i in range(2)
    ):
        raise ValueError("invalid independent or synchronized phase boundaries")
    bottles = [
        dict(
            object_id=i,
            robot_id=ARMS[i],
            approach_start=c[0][0],
            pickup_frame=c[0][1],
            grasp_frame=c[0][1],
            release_frame=c[1][0],
            retract_end=ends[i],
        )
        for i, c in enumerate(cycles)
    ]
    return dict(
        source_frames=n,
        fps=fps,
        shared_start=shared_start,
        shared_end=shared_end,
        prepare_start=starts,
        prepare_end=ends,
        return_end=returns,
        head_removed=min(starts),
        arm_head_removed=starts,
        cycles=cycles,
        tray_release_frames=[x[0] for x in release],
        bottle_events=bottles,
        duration_source="frozen_action_sequence",
        motion_threshold=motion_threshold,
        image_verified=False,
        boundary_policy="bottle opening +0.6s; later paired tray opening +0.2s",
    )


def phase_durations(phase, name):
    start = phase["prepare_start"] if name == "prepare" else [phase["shared_end"]] * 2
    end = phase["prepare_end"] if name == "prepare" else phase["return_end"]
    durations = np.asarray(end, dtype=int) - start
    if np.any(durations <= 0):
        raise ValueError("independent phases must have positive duration")
    return durations


def uniform_timing(phase, grid_index, grid_size):
    """Inclusive endpoints: left serial -> right serial, in signed frame time."""
    left, right = map(int, phase_durations(phase, "prepare"))
    segments = EpisodeSegments(0, ArmSegment(0, left), ArmSegment(0, right), 0, 0, 0, 0)
    plan = build_uniform_schedule_plan(segments, grid_index, grid_size)
    delta = plan.right_start_frame - plan.left_start_frame
    first = 0 if delta >= 0 else 1
    timing = Timing(ARMS[first], abs(delta) / (left, right)[first])
    return timing, dict(
        grid_index=grid_index,
        grid_size=grid_size,
        requested_offset_frames=left - grid_index * (left + right) / grid_size,
        signed_offset_frames=delta,
        endpoint_offsets_frames=[left, -right],
        quantization="nearest frame; at most half a frame",
    )


def independent_block(starts, ends, timing):
    """Clock samples include both endpoints; duration is samples minus one."""
    starts, ends = np.asarray(starts, dtype=int), np.asarray(ends, dtype=int)
    durations = ends - starts
    if np.any(durations <= 0):
        raise ValueError("empty independent motion")
    first = ARMS.index(timing.first_arm)
    delays = np.zeros(2, dtype=int)
    delays[1 - first] = math.floor(timing.delay_ratio * int(durations[first]) + 0.5)
    timeline = np.arange(int(np.max(delays + durations)) + 1)
    clocks = starts + np.clip(timeline[:, None] - delays, 0, durations)
    idle = timeline[:, None] < delays
    # The last hold sample borders an advancing command and remains supervised.
    advancing = np.zeros_like(idle)
    advancing[1:] |= np.diff(clocks, axis=0) > 0
    advancing[:-1] |= np.diff(clocks, axis=0) > 0
    idle &= ~advancing
    required = timeline[:, None] > delays + durations
    receipt = dict(
        **asdict(timing),
        duration_frames=durations.tolist(),
        onset_frames=delays.tolist(),
        finish_frames=(delays + durations).tolist(),
        signed_offset_frames=int(delays[1] - delays[0]),
        duration_source="frozen_action_sequence",
    )
    return clocks, idle, required, receipt


def join_to_source(starts, stop):
    """Supervised continuation after both arms finish, to a common source frame."""
    starts = np.asarray(starts, dtype=int)
    steps = np.arange(int(stop - starts.min()) + 1)[:, None]
    return np.minimum(starts + steps, stop)


def make_schedule(phase, timing, *, terminal_seconds=1.5):
    """Reuse one arm and p in both phases, including unequal arm durations."""
    n, fps = phase["source_frames"], phase["fps"]
    if not np.isfinite(terminal_seconds) or terminal_seconds < 0:
        raise ValueError("terminal_seconds must be finite and nonnegative")
    S, E = phase["shared_start"], phase["shared_end"]
    pre, pi, pw, pt = independent_block(
        phase["prepare_start"], phase["prepare_end"], timing
    )
    bridge = join_to_source(phase["prepare_end"], S)[1:]
    pre = np.vstack([pre, bridge])
    pi = np.vstack([pi, np.zeros_like(bridge, dtype=bool)])
    pw = np.vstack([pw, np.ones_like(bridge, dtype=bool)])
    middle = np.repeat(np.arange(S + 1, E + 1)[:, None], 2, axis=1)
    post, ri, rw, rt = independent_block([E, E], phase["return_end"], timing)
    rest = join_to_source(phase["return_end"], n - 1)[1:]
    post = np.vstack([post[1:], rest])
    ri = np.vstack([ri[1:], np.zeros_like(rest, dtype=bool)])
    rw = np.vstack([rw[1:], np.ones_like(rest, dtype=bool)])
    tail = max(0, math.ceil(terminal_seconds * fps) - 1)
    terminal = np.full((tail, 2), n - 1, dtype=int)
    clocks = np.vstack([pre, middle, post, terminal])
    idle = np.vstack(
        [pi, np.zeros_like(middle, dtype=bool), ri, np.zeros_like(terminal, dtype=bool)]
    )
    waits = np.vstack(
        [pw, np.zeros_like(middle, dtype=bool), rw, np.zeros_like(terminal, dtype=bool)]
    )
    stage = np.repeat([0, 1, 2, 3], [len(pre), len(middle), len(post), tail])
    # The coordinated interval begins once both clocks reach S, not when one
    # arm has already reached its endpoint while the other is still finishing.
    shared = (clocks[:, 0] == clocks[:, 1]) & (clocks[:, 0] >= S) & (clocks[:, 0] <= E)
    result = dict(
        clocks=clocks,
        active=(~idle).astype(np.float32),
        required_wait=waits,
        shared=shared,
        stage=stage,
        synthetic=stage == 3,
        timing=dict(**asdict(timing), prepare=pt, return_timing=rt),
    )
    validate_schedule(result, phase)
    return result


def validate_schedule(schedule, phase):
    clocks, active = schedule["clocks"], schedule["active"]
    n = phase["source_frames"]
    if clocks.ndim != 2 or clocks.shape[1] != 2 or active.shape != clocks.shape:
        raise ValueError("invalid clock/mask shape")
    delta = np.diff(clocks, axis=0)
    if (
        not np.isfinite(clocks).all()
        or clocks.min() < 0
        or clocks.max() >= n
        or np.any(delta < 0)
        or np.any(delta > 1)
    ):
        raise ValueError("source frames skipped or out of bounds")
    if not np.isin(active, [0, 1]).all():
        raise ValueError("active mask must be binary")
    if np.count_nonzero((active == 0).any(axis=0)) > 1:
        raise ValueError("an episode may mask only one delayed arm")
    S, E = phase["shared_start"], phase["shared_end"]
    shared = schedule["shared"]
    expected_shared = (
        (clocks[:, 0] == clocks[:, 1]) & (clocks[:, 0] >= S) & (clocks[:, 0] <= E)
    )
    interior = ((clocks > S) & (clocks < E)).any(axis=1)
    middle = np.repeat(np.arange(S + 1, E + 1)[:, None], 2, axis=1)
    if (
        not np.array_equal(shared, expected_shared)
        or not np.array_equal(clocks[interior, 0], clocks[interior, 1])
        or not np.array_equal(clocks[schedule["stage"] == 1], middle)
    ):
        raise ValueError("shared tray clocks diverge or omit synchronized frames")
    moving = np.zeros_like(active, dtype=bool)
    moving[1:] |= delta > 0
    moving[:-1] |= delta > 0
    if not np.all(active[moving] == 1):
        raise ValueError("motion onset incorrectly masked")
    if not np.all(active[shared] == 1) or not np.all(
        active[schedule["required_wait"]] == 1
    ):
        raise ValueError("required coordination masked")
    if (
        not np.all(active[clocks == n - 1] == 1)
        or not np.all(active[schedule["synthetic"]] == 1)
        or not np.all(clocks[-1] == n - 1)
    ):
        raise ValueError("task completion must remain supervised")
