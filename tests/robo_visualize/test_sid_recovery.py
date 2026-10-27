from robo_visualize.assets import assets_root
import importlib.util
from pathlib import Path

from robo_visualize.assets import assets_root
import importlib.util
import numpy as np
import pytest

from robo_visualize.arms.piperx.model import PiperXModel

ROOT = Path(__file__).parents[2]
SPEC = importlib.util.spec_from_file_location(
    "render_sid_recovery_svg",
    ROOT / "scripts/render_sid_recovery_svg.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
render = MODULE.render


def _payload() -> dict:
    pose_a = [0.0, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    pose_b = [0.2, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
    samples = []
    for index in range(10):
        x = 0.02 + index * 0.016
        y = (index - 4.5) * 0.012
        z = 0.18 + (index % 3) * 0.02
        p_pose = [x, y, z, 1.0, 0.0, 0.0, 0.0]
        q_pose = [x, 0.0, 0.2, 1.0, 0.0, 0.0, 0.0]
        endpoint = [x, y * 0.1, 0.2 + (z - 0.2) * 0.1, 1.0, 0.0, 0.0, 0.0]
        samples.append(
            {
                "sample_index": index,
                "p_pose_base_link6_wxyz": p_pose,
                "attached_episode_q_pose_base_link6_wxyz": q_pose,
                "predicted_endpoint_pose_base_link6_wxyz": endpoint,
                "metrics": {
                    "base_translation_direction_cosine": 0.99,
                    "nearest_manifold_distance_before": 2.0,
                    "nearest_manifold_distance_after": 0.2,
                    "nearest_manifold_distance_reduction_fraction": 0.9,
                },
            }
        )
    return {
        "schema": "robo_visualize.sid_recovery_vectors.v2",
        "coordinate_frame": "base_link",
        "controlled_frame": "link6",
        "expert_episode_index": 2,
        "manifold": {
            "full_trajectory_poses_base_link6_wxyz": [pose_a, pose_b],
            "approach_trajectory_poses_base_link6_wxyz": [pose_a, pose_b],
        },
        "sample_selection": {
            "seed": 20260828,
            "translation_radius_m": 0.1,
            "rotation_radius_rad": 0.35,
        },
        "samples": samples,
        "aggregate_metrics": {
            "sample_count": 10,
            "positive_direction_count": 10,
            "distance_reduced_count": 10,
            "mean_base_translation_direction_cosine": 0.99,
            "minimum_base_translation_direction_cosine": 0.98,
            "mean_nearest_manifold_distance_reduction_fraction": 0.9,
            "minimum_nearest_manifold_distance_reduction_fraction": 0.8,
        },
    }


def test_recovery_vectors_loader_and_cli_are_wired() -> None:
    server = (
        ROOT / "src/robo_visualize/arms/piperx/server.py"
    ).read_text()
    web_server = (ROOT / "src/robo_visualize/web/server.py").read_text()
    app = (ROOT / "src/robo_visualize/web/static/app.js").read_text()

    assert "def _load_recovery_vectors" in server
    assert '"--recovery-vectors"' in server
    assert 'payload.get("controlled_frame") != "link6"' in server
    assert "preferred_episode_index" in server
    assert '@app.get("/api/recovery_vectors.json")' in web_server
    assert 'sampleGroup.name = "sid-recovery-sample-"' in app
    assert 'appendOverlayLegend("P1–P"' in app
    assert 'appendOverlayLegend("actual EE TCP trajectory"' in app
    assert "composePoses(pose, controlledToDisplay)" in app
    assert '"has_ee_trajectory": True' in server
    assert 'list(self.model.relative_frame_pose("link6", "tcp"))' in server
    assert 'payload["ee_trajectory_url"]' in server
    assert "backend.ee_trajectory(dataset, episode_index, signal)" in web_server
    assert "fetchChecked(signalUrl(episode.ee_trajectory_url)" in app


def test_static_renderer_rejects_wrong_schema() -> None:
    payload = _payload()
    payload["schema"] = "wrong"

    with pytest.raises(ValueError, match="schema"):
        render(payload)


def test_static_svg_contains_ten_labeled_recovery_directions() -> None:
    svg = render(_payload())

    assert "PiperX SID Recovery Field" in svg
    assert "10 unfiltered Omega samples" in svg
    assert "P1" in svg
    assert "P10" in svg
    assert svg.count('marker-end="url(#predicted-head)"') == 20
    assert svg.count('marker-end="url(#reference-head)"') == 20
    assert "distance reduced 10/10" in svg


@pytest.mark.skipif(importlib.util.find_spec("pinocchio") is None or not (assets_root() / "verified-assets.json").is_file(), reason="requires PiperX geometry runtime")
def test_recorded_arm_angles_are_not_clipped_to_nominal_urdf_limits() -> None:
    model = PiperXModel()
    state = np.zeros(7, dtype=np.float64)
    state[2] = 1.138

    model.update(state)

    joint_index = model.joint_q_indices["joint3"]
    assert model.q[joint_index] == pytest.approx(np.deg2rad(state[2]))
    assert model.q[joint_index] > model.model.upperPositionLimit[joint_index]


@pytest.mark.skipif(importlib.util.find_spec("pinocchio") is None or not (assets_root() / "verified-assets.json").is_file(), reason="requires PiperX geometry runtime")
def test_link6_to_tcp_pose_matches_the_fixed_gripper_offset() -> None:
    model = PiperXModel()

    pose = model.relative_frame_pose("link6", "tcp")

    np.testing.assert_allclose(pose[:3], [0.0, 0.0, 0.1358], atol=1e-9)
    np.testing.assert_allclose(pose[3:], [1.0, 0.0, 0.0, 0.0], atol=1e-9)
