"""Plan or render water timing with one first arm and delay ratio per episode."""

import argparse
import json
from pathlib import Path

from .timeline.water import Timing, detect_phases, make_schedule, uniform_timing
from .trim import read_episode, source_episodes
from .water_io import dump, finish_export, render_source, source_view


def plan_source(table, fps, source_index, source_count, samples, timing, threshold):
    phase = detect_phases(
        table["action"].to_pylist(),
        table["observation.state"].to_pylist(),
        fps,
        motion_threshold=threshold,
    )
    schedules = []
    for variant in range(samples if timing is None else 1):
        grid = None
        selected = timing
        if selected is None:
            # Interleaved global grid, including both serial endpoints.
            index = source_index + source_count * variant
            selected, grid = uniform_timing(phase, index, source_count * samples - 1)
        schedule = make_schedule(phase, selected)
        schedule["uniform"] = grid
        schedules.append(schedule)
    return phase, schedules


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["plan", "render"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New output directory"
    )
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument(
        "--samples", type=int, default=4, help="Uniform samples per source"
    )
    parser.add_argument("--first-arm", choices=["left", "right"])
    parser.add_argument(
        "--delay-ratio", type=float, help="p in [0,1]; 1 is strictly serial"
    )
    parser.add_argument(
        "--motion-threshold", type=float, default=3.0, help="degrees/s and mm/s"
    )
    args = parser.parse_args(argv)
    if (args.first_arm is None) != (args.delay_ratio is None):
        parser.error("--first-arm and --delay-ratio must be supplied together")
    timing = (
        Timing(args.first_arm, args.delay_ratio) if args.first_arm is not None else None
    )
    if timing is None and args.samples < 2:
        parser.error("uniform sampling needs at least two samples per source")
    source = args.source.resolve()
    args.output = args.output.resolve()
    if args.output.resolve() == source or source in args.output.resolve().parents:
        parser.error("output must be outside the source dataset")
    info = json.loads((source / "meta/info.json").read_text())
    rows = source_episodes(source)
    if args.episodes is not None:
        requested = set(args.episodes)
        if not requested <= {r["episode_index"] for r in rows}:
            parser.error("unknown source episode")
        rows = [r for r in rows if r["episode_index"] in requested]
    args.output.mkdir(parents=True, exist_ok=False)
    plans, checks = [], []
    for index, row in enumerate(rows):
        table = read_episode(source, info, row)
        phase, schedules = plan_source(
            table,
            info["fps"],
            index,
            len(rows),
            args.samples,
            timing,
            args.motion_threshold,
        )
        ep = row["episode_index"]
        record = dict(
            source_episode=ep,
            phase=phase,
            variants=[
                dict(
                    timing=s["timing"],
                    uniform=s["uniform"],
                    frames=len(s["clocks"]),
                    idle_frames=(s["active"] == 0).sum(axis=0).tolist(),
                )
                for s in schedules
            ],
        )
        plans.append(record)
        dump(args.output / "plans.json", plans)
        print(
            f"source {ep}: {len(schedules)} samples; head removed {phase['head_removed']}",
            flush=True,
        )
        if args.mode == "render":
            from .interaction.video import read_video
            from .water_segmentation import segment

            work = args.output / "work" / f"source-{ep:03d}"
            work.mkdir(parents=True)
            view, _, _ = source_view(source, info, row, work)
            frames, _ = read_video(
                view / "videos/observation.images.top/chunk-000/file-000.mp4"
            )
            segment(work, frames, phase)
            del frames
            checks.extend(
                render_source(
                    view,
                    work,
                    args.output / "dataset",
                    ep,
                    phase,
                    schedules,
                    len(checks),
                )
            )
    if args.mode == "render":
        print(finish_export(view, args.output / "dataset", checks))


if __name__ == "__main__":
    main()
