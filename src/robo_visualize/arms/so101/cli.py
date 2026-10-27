from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from robo_visualize.arms.so101.checker import SO101CollisionChecker
from robo_visualize.arms.so101.trajectory import SO101Trajectory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Mesh-BVH collision analysis and 3D rendering for dual SO101 trajectories.",
    )
    parser.add_argument("parquet", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument(
        "--signal",
        choices=("action", "observation.state"),
        default="action",
    )
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--interpolation-substeps", type=int, default=1)
    parser.add_argument("--safety-margin-m", type=float, default=0.0)
    parser.add_argument("--compute-minimum-distance", action="store_true")
    parser.add_argument("--collect-all-pairs", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--render-video", type=Path)
    parser.add_argument("--render-stride", type=int, default=1)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=720)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    trajectory = SO101Trajectory.from_lerobot_parquet(
        args.parquet,
        args.episode,
        args.signal,
    )
    if args.seconds is not None:
        trajectory = trajectory.first_seconds(args.seconds)
    trajectory = trajectory.interpolated(args.interpolation_substeps)
    checker = SO101CollisionChecker(safety_margin_m=args.safety_margin_m)
    report = checker.analyze_trajectory(
        trajectory,
        compute_minimum_distance=args.compute_minimum_distance,
        collect_all_pairs=args.collect_all_pairs,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report.to_dict(), indent=2))
    if args.render_video:
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")
        from robo_visualize.arms.so101.render import SO101Renderer

        renderer = SO101Renderer(
            checker,
            width=args.render_width,
            height=args.render_height,
        )
        renderer.render_trajectory(
            trajectory,
            args.render_video,
            stride=args.render_stride,
        )
    print(json.dumps(summary_for(report, args), indent=2))


def summary_for(report, args: argparse.Namespace) -> dict:
    return {
        "source": report.source,
        "frames": report.frame_count,
        "duration_s": report.duration_s,
        "collision_frames": report.collision_frames,
        "collision_rate": report.collision_rate,
        "first_collision_frame": report.first_collision_frame,
        "minimum_distance_m": report.minimum_distance_m,
        "setup_seconds": report.setup_seconds,
        "analysis_seconds": report.analysis_seconds,
        "frames_per_second": report.frames_per_second,
        "report": str(args.report),
        "render_video": str(args.render_video) if args.render_video else None,
    }


if __name__ == "__main__":
    main()
