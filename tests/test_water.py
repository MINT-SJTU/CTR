import numpy as np
import pytest

from real_robot_data_retime.timeline.water import (
    Timing,
    detect_phases,
    independent_block,
    make_schedule,
    uniform_timing,
    validate_schedule,
)


def phase():
    return dict(
        source_frames=200,
        fps=30,
        shared_start=70,
        shared_end=120,
        prepare_start=[10, 15],
        prepare_end=[70, 55],
        return_end=[170, 190],
        head_removed=10,
        arm_head_removed=[10, 15],
    )


@pytest.mark.parametrize("arm", ["left", "right"])
@pytest.mark.parametrize("p", [0, 0.25, 0.5, 1])
def test_same_delayed_arm_for_both_phases(arm, p):
    result = make_schedule(phase(), Timing(arm, p))
    c, a = result["clocks"], result["active"]
    assert (c[0] == [10, 15]).all()
    assert ((a == 0).any(axis=0)).sum() <= 1
    assert (a[:, 0 if arm == "left" else 1] == 1).all()
    assert (a[-45:] == 1).all()
    assert (c[-45:] == 199).all()
    assert (a[result["required_wait"]] == 1).all()
    np.testing.assert_array_equal(c[result["shared"], 0], c[result["shared"], 1])
    for key in ("prepare", "return_timing"):
        receipt = result["timing"][key]
        assert receipt["first_arm"] == arm
        assert receipt["delay_ratio"] == p
        first = 0 if arm == "left" else 1
        if p == 1:
            assert receipt["onset_frames"][1 - first] == receipt["finish_frames"][first]
    if p == 0:
        assert (a == 1).all()


def test_signed_uniform_grid_with_unequal_durations_and_serial_endpoints():
    timings = [uniform_timing(phase(), i, 10) for i in range(11)]
    assert [r["signed_offset_frames"] for _, r in timings] == list(range(60, -41, -10))
    assert timings[0][0] == Timing("left", 1)
    assert timings[-1][0] == Timing("right", 1)
    for timing, _ in timings:
        result = make_schedule(phase(), timing)
        assert (
            result["timing"]["prepare"]["delay_ratio"]
            == result["timing"]["return_timing"]["delay_ratio"]
        )


def test_serial_never_advances_follower_before_leader_finishes():
    clocks, idle, _, receipt = independent_block([10, 15], [70, 55], Timing("left", 1))
    leader_done = np.flatnonzero(clocks[:, 0] == 70)[0]
    follower_move = np.flatnonzero(np.diff(clocks[:, 1]) > 0)[0] + 1
    assert follower_move > leader_done
    assert receipt["duration_frames"] == [60, 40]
    assert not idle[leader_done, 1]  # Supervise the onset edge.


@pytest.mark.parametrize("fps", [20, 30, 50])
def test_release_boundaries_use_seconds_and_trim_frozen_prefix(fps):
    n = fps * 10
    action = np.zeros((n, 14))
    for side in range(2):
        offset = side * 7
        action[fps : n - fps, offset] = np.arange(n - 2 * fps)
        action[n - fps :, offset] = n - 2 * fps - 1
        for begin, end in [(1, 2), (3, 5), (7, 8)]:
            action[begin * fps : end * fps, offset + 6] = 40
    result = detect_phases(action, action, fps)
    assert result["prepare_end"] == [3 * fps + int(0.6 * fps)] * 2
    assert result["shared_end"] == 7 * fps + int(0.2 * fps)
    assert result["head_removed"] > 0
    assert result["shared_end"] < 8 * fps  # Do not wait for the empty reclose.
    assert all(e["release_frame"] == 3 * fps for e in result["bottle_events"])


def test_episode_wide_mask_guard_and_invalid_inputs():
    result = make_schedule(phase(), Timing("left", 1))
    result["active"][0, 0] = 0
    with pytest.raises(ValueError, match="only one"):
        validate_schedule(result, phase())
    for p in [-0.1, 1.1, float("nan")]:
        with pytest.raises(ValueError):
            Timing("left", p)
    with pytest.raises(ValueError):
        uniform_timing(phase(), 11, 10)


def test_sync_validation_does_not_trust_shared_labels():
    result = make_schedule(phase(), Timing("left", 0))
    result["shared"][:] = False
    with pytest.raises(ValueError, match="shared"):
        validate_schedule(result, phase())



@pytest.mark.parametrize("fps", [20, 30, 50])
def test_tray_regrasp_stays_synchronized_and_empty_reopen_does_not_extend_it(fps):
    n = 15 * fps
    action = np.zeros((n, 14))
    for side in range(2):
        offset = side * 7
        action[:, offset] = np.arange(n)
        # Bottle approach/release, short failed grip, actual carry, empty reopen.
        for begin, end in [(1, 2), (3, 4), (5, 6), (10, 11), (12, 13)]:
            action[begin * fps:end * fps, offset + 6] = 40
    result = detect_phases(action, action, fps)
    assert result["tray_release_frames"] == [10 * fps, 10 * fps]
    assert result["shared_end"] == 10 * fps + int(.2 * fps)
    for side in ("left", "right"):
        schedule = make_schedule(result, Timing(side, 1))
        carry = schedule["clocks"][:, 0]
        carry = (carry >= 6 * fps) & (carry <= 10 * fps)
        assert (schedule["clocks"][carry, 0] == schedule["clocks"][carry, 1]).all()
