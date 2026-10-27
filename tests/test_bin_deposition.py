import numpy as np
from real_robot_data_retime.tasks.workpiece import verify_deposit, bin_visits


def test_empty_visit_is_not_a_deposit():
    frames = np.full((80, 80, 100, 3), 180, np.uint8)
    robots = np.zeros((80, 80, 100), bool)
    visit = dict(clearance_frame=40, release_frame=35)
    empty = verify_deposit(frames, robots, 20, visit, [10, 10, 40, 40], 30)
    assert not empty["verified"]
    frames[35:, 25:33, 25:33] = 30
    assert verify_deposit(frames, robots, 20, visit, [10, 10, 40, 40], 30)["verified"]


def test_bin_visit_requires_dwell_then_exit():
    xy = np.tile([80.0, 70.0], (90, 1))
    xy[30:50] = [30, 30]
    visit = next(
        bin_visits(xy, 10, 90, [10, 10, 40, 40], 30, robot_coverage=np.zeros(90))
    )
    assert visit["entry_frame"] == 30 and visit["clearance_frame"] == 50


def test_final_withdrawal_can_leave_camera_after_observed_bin_coverage():
    xy = np.tile([80.0, 70.0], (90, 1))
    xy[30:50] = [30, 30]
    xy[50:] = np.nan
    robots = np.zeros((90, 80, 100), bool)
    robots[30:55, 10:50, 10:50] = True
    visit = next(
        bin_visits(
            xy,
            10,
            90,
            [10, 10, 40, 40],
            30,
            robot_coverage=robots[:, 10:50, 10:50].mean(axis=(1, 2)),
        )
    )
    assert visit["clearance_frame"] == 55
    assert (
        list(bin_visits(xy, 10, 90, [10, 10, 40, 40], 30, robot_coverage=np.zeros(90)))
        == []
    )


def test_arm_overlap_proposes_bin_visit_when_tip_is_clipped():
    xy = np.full((90, 2), np.nan)
    coverage = np.zeros(90)
    coverage[30:55] = 0.4
    visit = next(bin_visits(xy, 10, 90, [10, 10, 40, 40], 30, robot_coverage=coverage))
    assert visit["entry_frame"] == 30
    assert visit["clearance_frame"] == 55
    frames = np.full((90, 80, 100, 3), 180, np.uint8)
    assert not verify_deposit(
        frames, np.zeros((90, 80, 100), bool), 10, visit, [10, 10, 40, 40], 30
    )["verified"]


def test_all_visit_hypotheses_are_available_for_verification():
    xy = np.tile([90.0, 75.0], (100, 1))
    xy[25:40] = [30, 30]
    xy[55:70] = [30, 30]
    visits = list(
        bin_visits(xy, 10, 100, [10, 10, 40, 40], 30, robot_coverage=np.zeros(100))
    )
    assert [v["entry_frame"] for v in visits] == [25, 55]


def test_a_later_deposit_cannot_verify_an_empty_earlier_visit():
    frames = np.full((100, 80, 100, 3), 180, np.uint8)
    robots = np.zeros((100, 80, 100), bool)
    robots[42:65, 10:50, 10:50] = True
    frames[65:, 25:33, 25:33] = 30
    result = verify_deposit(
        frames,
        robots,
        20,
        dict(clearance_frame=40, release_frame=35),
        [10, 10, 40, 40],
        30,
    )
    assert not result["verified"]


def test_reference_precedes_interaction_not_delayed_visible_attachment():
    frames = np.full((80, 80, 100, 3), 180, np.uint8)
    robots = np.zeros((80, 80, 100), bool)
    frames[35:, 25:33, 25:33] = 30
    visit = dict(clearance_frame=45, release_frame=42)
    assert verify_deposit(frames, robots, 20, visit, [10, 10, 40, 40], 30)["verified"]
    assert not verify_deposit(frames, robots, 40, visit, [10, 10, 40, 40], 30)[
        "verified"
    ]
