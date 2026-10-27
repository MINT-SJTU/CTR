import numpy as np
import pytest

from real_robot_data_retime.timeline.terminal import terminal_hold_plan


def test_short_recording_and_existing_rest():
    moving = np.repeat(np.arange(10)[:, None], 14, axis=1).astype(float)
    for rest in [0, 10, 45, 60]:
        x = np.concatenate([moving, np.repeat(moving[-1:], rest, axis=0)])
        p = terminal_hold_plan(x, x, 30)
        assert p["recorded_static_frames"] == rest
        assert p["added_frames"] == max(0, 45 - rest)


def test_both_state_and_action_must_stop():
    state = np.zeros((60, 14))
    action = state.copy()
    state[-10:, 7] = np.arange(10)
    assert terminal_hold_plan(state, action, 30)["added_frames"] == 45
    action[-10:, 0] = np.arange(10)
    assert terminal_hold_plan(np.zeros_like(state), action, 30)["added_frames"] == 45


def test_invalid_and_single_sample():
    assert (
        terminal_hold_plan(np.zeros((1, 14)), np.zeros((1, 14)), 30)["added_frames"]
        == 45
    )
    with pytest.raises(ValueError):
        terminal_hold_plan(np.zeros((2, 14)), np.zeros((3, 14)), 30)


def test_video_extension_checks_original_frames_and_cloned_tail(tmp_path):
    import importlib.util
    import shutil
    import subprocess
    from pathlib import Path

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg and ffprobe required")
    script = Path(__file__).parents[1] / "scripts/complete_screw_terminal_hold.py"
    spec = importlib.util.spec_from_file_location("terminal_export", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = tmp_path / "original.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=64x48:rate=30",
            "-frames:v",
            "10",
            "-c:v",
            "libx264",
            str(original),
        ],
        check=True,
    )
    _, error = module.extend_video(original, tmp_path / "extended.mp4", 10, 45, 30)
    assert 0 <= error <= 8
    module.extend_video(
        original, tmp_path / "reused.mp4", 10, 45, 30, encoded=tmp_path / "extended.mp4"
    )
    assert (tmp_path / "reused.mp4").read_bytes() == (
        tmp_path / "extended.mp4"
    ).read_bytes()
    module.extend_video(original, tmp_path / "unchanged.mp4", 10, 0, 30)
    assert original.read_bytes() == (tmp_path / "unchanged.mp4").read_bytes()
