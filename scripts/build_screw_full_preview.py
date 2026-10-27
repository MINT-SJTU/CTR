"""Publish explicit 25-episode coverage, including incomplete jobs."""

import argparse
import json
from pathlib import Path


def build(root, configs):
    records = []
    cards = []
    for ep in range(25):
        folder = root / f"ep{ep:03d}"
        path = folder / "cases.json"
        complete = False
        frames = []
        config = json.loads((configs / f"ep{ep:03d}.json").read_text())
        if path.exists():
            cases = json.loads(path.read_text())
            complete = len(cases) == 3 and all(
                (folder / c["name"] / "report.json").exists() for c in cases
            )
            if complete:
                for case in cases:
                    report = json.loads(
                        (folder / case["name"] / "report.json").read_text()
                    )
                    complete &= (
                        report["validation"]["passed"]
                        and report["plan"]["validation"]["passed"]
                        and report["inputs"]["episode_index"] == ep
                        and report["inputs"]["config"] == config
                    )
                    complete &= all(
                        (folder / case["name"] / name).is_file()
                        and (folder / case["name"] / name).stat().st_size > 0
                        for name in [
                            "parallel.mp4",
                            "preview.webm",
                            "left_wrist.mp4",
                            "right_wrist.mp4",
                            "trajectories.parquet",
                            "source_mapping.npz",
                        ]
                    )
                    frames.append(case["frames"])
        records.append(
            {
                "episode": ep,
                "complete": bool(complete),
                "videos": 3 if complete else 0,
                "frames": frames,
            }
        )
        cards.append(
            f'<a href="ep{ep:03d}/">Episode {ep} · 3 种时序</a>'
            if complete
            else f'<div class="pending">Episode {ep} · 处理中</div>'
        )
    count = sum(r["complete"] for r in records)
    receipt = {
        "expected_episodes": 25,
        "completed_episodes": count,
        "completed_videos": count * 3,
        "complete": count == 25,
        "episodes": records,
    }
    (root / "coverage.json").write_text(json.dumps(receipt, indent=2) + "\n")
    idle_link = (
        '<p><a href="idle-review/">打开三视角审核与 Idle Mask</a></p>'
        if (root / "idle-review/index.html").exists()
        else ""
    )
    (root / "index.html").write_text(
        """<!doctype html><html lang="zh"><meta charset="utf-8"><title>插螺丝 · 全数据集</title><style>body{background:#11161c;color:#eef2f6;font:18px system-ui;max-width:1100px;margin:36px auto;padding:24px}nav{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}a,.pending{display:block;border-radius:10px;padding:18px;background:#202a35;color:#9fd4ff;text-decoration:none}.pending{color:#929daa}p{line-height:1.8}</style><h1>插螺丝 · 全数据集</h1>"""
        + idle_link
        + f"<p>已完成 {count} / 25 个 episode · {count * 3} / 75 条时序视频</p><p>每条示范提供三种时序、主视角及双腕视频和同步运动数据。自动修复无法确认的疑似遮挡可在视频下方跳转复核。</p><nav>"
        + "".join(cards)
        + "</nav></html>"
    )
    return receipt


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--configs", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.configs)))
