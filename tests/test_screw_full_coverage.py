import importlib.util
import json
from pathlib import Path


def builder():
    path = Path(__file__).parents[1] / "scripts/build_screw_full_preview.py"
    spec = importlib.util.spec_from_file_location("full_preview", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build


def test_full_completion_requires_all_25_episodes_and_valid_outputs(tmp_path):
    root = tmp_path / "outputs"
    configs = tmp_path / "configs"
    root.mkdir()
    configs.mkdir()
    for ep in range(25):
        config = {"episode_index": ep}
        (configs / f"ep{ep:03d}.json").write_text(json.dumps(config))
        folder = root / f"ep{ep:03d}"
        folder.mkdir()
        cases = []
        for name in ["screw-15", "screw-50", "screw-85"]:
            case = folder / name
            case.mkdir()
            cases.append({"name": name, "frames": 10})
            for file in [
                "parallel.mp4",
                "preview.webm",
                "left_wrist.mp4",
                "right_wrist.mp4",
                "trajectories.parquet",
                "source_mapping.npz",
            ]:
                (case / file).write_bytes(b"x")
            (case / "report.json").write_text(
                json.dumps(
                    {
                        "validation": {"passed": True},
                        "plan": {"validation": {"passed": True}},
                        "inputs": {"episode_index": ep, "config": config},
                    }
                )
            )
        (folder / "cases.json").write_text(json.dumps(cases))
    build = builder()
    receipt = build(root, configs)
    assert receipt["complete"] and receipt["completed_videos"] == 75
    (root / "ep024/screw-85/right_wrist.mp4").unlink()
    receipt = build(root, configs)
    assert not receipt["complete"] and receipt["completed_episodes"] == 24
    (configs / "ep003.json").write_text(
        json.dumps({"episode_index": 3, "new_annotation": True})
    )
    assert build(root, configs)["completed_episodes"] == 23
