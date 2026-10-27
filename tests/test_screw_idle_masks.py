import json
from pathlib import Path

import numpy as np
import pytest

from real_robot_data_retime.timeline.idle import (
    arm_active_mask,
    clock_idle_mask,
    screw_idle_masks,
    terminal_rest_limit,
)
from real_robot_data_retime.timeline.screw import screw_schedule
from real_robot_data_retime.timeline.smooth import sample_rows


def test_clock_mask_retains_departure_and_terminal_arrival():
    np.testing.assert_array_equal(
        clock_idle_mask([0, 0, 1, 1, 2, 2, 2]),
        [True, False, True, False, False, True, True],
    )
    assert not clock_idle_mask([0, 1e-10, 1]).any()
    assert clock_idle_mask([1, 1, 1]).all()
    with pytest.raises(ValueError):
        clock_idle_mask([1, 0])


def test_terminal_rest_does_not_bridge_late_return_motion():
    state = np.zeros((120, 7))
    state[:60, 0] = 2
    state[60:70, 0] = np.linspace(2, 0, 10)
    r = terminal_rest_limit(state, state, 0, 30)
    assert r["start"] >= 68 and r["mask_start"] == r["start"] + 45


@pytest.mark.parametrize("position", [0.15, 0.5, 0.85])
def test_screw_required_waits_insertion_and_retreat_remain_supervised(
    position, tmp_path
):
    root = Path(__file__).parents[1]
    c = json.loads((root / "tests/fixtures/screw-full-ep002.json").read_text())
    with np.load(root / "tests/fixtures/screw_ep002.npz") as d:
        state, action = d["state"], d["action"]
        left, right, plan = screw_schedule(
            state,
            action,
            int(d["start"]),
            int(d["stop"]),
            c["coupled_intervals"],
            c["ready_frames"],
            c["right_retreat_ends"],
            position,
            30,
        )
    values = [
        np.c_[sample_rows(v[:, :7], left), sample_rows(v[:, 7:], right)]
        for v in [state, action]
    ]
    result = screw_idle_masks(left, right, *values, plan["stages"], 30)
    masks = []
    for arm in ["left", "right"]:
        mask = np.zeros(len(left), bool)
        for a, b in result[arm]:
            mask[a:b] = True
        masks.append(mask)
        for a, b in result[arm + "_required_wait"]:
            assert not mask[a:b].any()
    active = arm_active_mask(result, len(left))
    assert active.dtype == np.float32
    import pyarrow as pa
    import pyarrow.parquet as pq

    from real_robot_data_retime.screw import label_trajectories
    from real_robot_data_retime.staged import export_trajectories

    export_trajectories(tmp_path, (state, action), left, right, 30, {"stages": {}})
    label_trajectories(tmp_path, left, right, plan, 30)
    table = pq.read_table(tmp_path / "trajectories.parquet")
    assert table["observation.arm_active_mask"].type == pa.list_(pa.float32(), 2)
    np.testing.assert_array_equal(
        table["observation.arm_active_mask"].to_pylist(), active
    )

    np.testing.assert_array_equal(active, 1 - np.column_stack(masks).astype(np.float32))
    assert any(m.any() for m in masks)
    for s in plan["stages"]:
        a, b = s["output_start"], s["output_end"] + 1
        if s["kind"] == "coupled":
            assert not masks[0][a:b].any() and not masks[1][a:b].any()
        if "mandatory_previous_right_retreat" in s:
            r = s["mandatory_previous_right_retreat"]
            if r["previous_cycle"] > 0:
                assert not masks[1][r["output_start"] : r["output_end"] + 1].any()
        if s["kind"] == "independent":
            for side in range(2):
                assert not masks[side][s["pickup_start_output_frames"][side] : b].any()


def test_arm_mask_order_boundaries_and_both_masked():
    result = arm_active_mask({"left": [[0, 2], [4, 5]], "right": [[1, 3]]}, 5)
    np.testing.assert_array_equal(result, [[0, 1], [0, 0], [1, 0], [1, 1], [0, 1]])
    for spans in ([[0, 6]], [[2, 3], [1, 2]], [[2, 2]]):
        with pytest.raises(ValueError, match="invalid idle intervals"):
            arm_active_mask({"left": spans, "right": []}, 5)
