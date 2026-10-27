import json
from pathlib import Path

import numpy as np
import pytest

from robo_visualize.assets import assets_root
from robo_visualize.arms.piperx.live_server import LiveViewerBackend, PiperXLiveModel


def _write_expert(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "robo_visualize.piperx_live_expert_trajectory.v1",
                "frame_indices": [10, 20],
                "link6_poses_base_wxyz": [
                    [0.2, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0],
                    [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0],
                ],
                "source": {"variant_id": "optimized"},
            }
        )
    )


def _write_state(path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "up_vla.sid_live_visualization.v2",
                "wall_time_unix_s": 100.0,
                "monotonic_s": 12.0,
                "attempt_index": 4,
                "phase": "planning",
                "decision": "recover",
                "current_action_deg_mm": [0.0] * 7,
                "current_link6_pose_base_wxyz": [
                    0.55767,
                    0.0,
                    0.19986,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "recovery_twist_link6_local": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
                "predicted_endpoint_link6_pose_base_wxyz": [
                    0.56767,
                    0.0,
                    0.19986,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                "nearest_expert_frame": 20,
                "distance_before": 0.7,
                "distance_after": 0.2,
                "recovery_takeover_generation": 2,
                "takeover": {"active": False, "generation": 2},
            }
        )
    )


@pytest.mark.skipif(not (assets_root() / "verified-assets.json").is_file(), reason="requires downloaded PiperX assets")
def test_live_backend_exposes_robot_expert_and_recovery_state(tmp_path) -> None:
    expert = tmp_path / "expert.json"
    state = tmp_path / "state.json"
    _write_expert(expert)
    _write_state(state)
    backend = LiveViewerBackend(state, expert)

    config = backend.config()
    payload = backend.live_state()

    assert config["live_mode"] is True
    assert config["mesh_count"] == 10
    assert len(backend.transforms("live", 0)) == 10 * 16 * 4
    assert config["live_stale_after_s"] == 2.0
    assert backend.expert_source == {"variant_id": "optimized"}
    assert len(backend.ee_trajectory("live", 0)) == 2 * 3 * 4
    assert len(payload["mesh_transforms_column_major"]) == 10
    assert payload["phase"] == "planning"
    assert payload["nearest_expert_frame"] == 20
    np.testing.assert_allclose(
        payload["current_tcp_pose_base_wxyz"][:3],
        [0.55767, 0.0, 0.33566],
    )
    np.testing.assert_allclose(
        payload["predicted_endpoint_tcp_pose_base_wxyz"][:3],
        [0.56767, 0.0, 0.33566],
    )
    np.testing.assert_allclose(
        payload["nearest_expert_tcp_pose_base_wxyz"][:3],
        [0.3, 0.0, 0.5358],
    )


@pytest.mark.skipif(not (assets_root() / "verified-assets.json").is_file(), reason="requires downloaded PiperX assets")
def test_live_model_returns_one_transform_per_checked_in_mesh() -> None:
    assets = (
        assets_root()
    )
    model = PiperXLiveModel(assets)

    action = np.zeros(7, dtype=np.float64)
    transforms = model.transforms(action)
    link6_pose = model.link6_pose_base_wxyz(action)

    np.testing.assert_allclose(
        link6_pose[:3],
        [0.097829053, 0.000001109, 0.216958021],
        atol=1e-5,
    )
    assert np.linalg.norm(link6_pose[3:]) == pytest.approx(1.0)
    assert len(transforms) == 10
    for transform in transforms:
        assert transform.shape == (4, 4)
        np.testing.assert_allclose(transform[3], [0.0, 0.0, 0.0, 1.0])
