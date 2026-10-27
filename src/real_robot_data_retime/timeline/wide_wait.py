"""Search recorded outside parking poses for exceptional workpiece episodes."""

from itertools import product

from .scheduler import NoSafeSchedule
from .workpiece_workspace import (
    DEFAULT_WORKSPACE,
    plan_workspace,
    workspace_waiting_stops,
)


def search_wide_waits(timeline, joints, cache):
    """Try ordinary entry first, then farther recorded poses and unlimited waits.

    The shared planner certifies every interpolated edge, retains preparation,
    and prohibits interruptions during admitted manipulations.
    """
    for padding in (0.0, 0.04, 0.09, 0.12):
        base = dict(DEFAULT_WORKSPACE, waiting_padding_m=padding)
        try:
            automatic = workspace_waiting_stops(cache, base)
        except ValueError as exc:
            if str(exc) != "no outside approach for fixed-volume waiting pose":
                raise
        else:
            break
    else:
        raise NoSafeSchedule("no outside waiting pose in expanded volumes")
    upper = [automatic[0][0], *automatic[1]]
    lower = [
        cache["own"][0][0]["release_frame"] + 1,
        cache["starts"][1],
        cache["own"][1][0]["release_frame"] + 1,
    ]
    candidates = []
    for lo, hi in zip(lower, upper):
        candidates.append(
            list(
                dict.fromkeys(
                    [
                        hi,
                        lo,
                        *[min(hi, lo + d) for d in (1, 2, 4, 8)],
                        *[lo + (hi - lo) * part // 4 for part in (1, 2, 3)],
                    ]
                )
            )
        )
    attempts = 0
    for values in product(*candidates):
        config = dict(base, waiting_source_frames=list(values), allow_initial_wait=True)
        try:
            workspace_waiting_stops(cache, config)
            left, right, plan = plan_workspace(timeline, joints, config, cache=cache)
        except NoSafeSchedule:
            attempts += 1
            continue
        except ValueError as exc:
            if str(exc) not in {
                "explicit wait is not on the outside preparation path",
                "no braking path after the protected execution",
                "outside stopping ramp interrupts preceding placement",
            }:
                raise
            attempts += 1
            continue
        return config, left, right, plan, attempts + 1
    raise NoSafeSchedule(
        f"no certified schedule in {attempts} expanded parking combinations"
    )
