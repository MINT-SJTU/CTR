import pytest

from real_robot_data_retime.timeline import wide_wait
from real_robot_data_retime.timeline.scheduler import NoSafeSchedule


def test_expanded_search_reaches_home_and_immediate_post_release(monkeypatch):
    cache = {
        "own": [[{"release_frame": 20}], [{"release_frame": 60}]],
        "starts": [0, 40],
    }
    monkeypatch.setattr(
        wide_wait, "workspace_waiting_stops", lambda c, cfg: [[30], [50, 70]]
    )
    tried = []

    def plan(timeline, joints, config, *, cache):
        stops = config["waiting_source_frames"]
        tried.append(stops)
        if stops != [21, 40, 61]:
            raise NoSafeSchedule("blocked")
        assert config["allow_initial_wait"]
        assert config["minimum_clearance_m"] == 0.05
        return "left", "right", {"certified": True}

    monkeypatch.setattr(wide_wait, "plan_workspace", plan)
    config, left, right, report, count = wide_wait.search_wide_waits({}, (), cache)
    assert (left, right) == ("left", "right")
    assert tried[0] == [30, 50, 70]
    assert config["waiting_source_frames"] == [21, 40, 61]
    assert count == len(tried)
    assert report["certified"]


def test_unexpected_errors_are_not_reclassified_as_unschedulable(monkeypatch):
    cache = {
        "own": [[{"release_frame": 20}], [{"release_frame": 60}]],
        "starts": [0, 40],
    }
    monkeypatch.setattr(
        wide_wait, "workspace_waiting_stops", lambda c, cfg: [[30], [50, 70]]
    )

    def fail(*args, **kwargs):
        raise ValueError("bad calibration")

    monkeypatch.setattr(wide_wait, "plan_workspace", fail)
    with pytest.raises(ValueError, match="bad calibration"):
        wide_wait.search_wide_waits({}, (), cache)


def test_home_hold_preserves_full_preparation_clock():
    import numpy as np

    from real_robot_data_retime.timeline.workpiece import approach_clock

    clock, holds, _ = approach_clock(40, 100, [40, 80], 30, brake_after={80: 70})
    assert holds[0] == 0
    delayed = np.r_[np.repeat(clock[0], 25), clock]
    assert delayed[0] == 40
    assert delayed[-1] == 100
    assert np.all(np.diff(delayed) >= 0)
    assert np.max(np.diff(delayed)) <= 1 + 1e-8
    assert np.flatnonzero(np.diff(delayed) > 0)[0] == 25
