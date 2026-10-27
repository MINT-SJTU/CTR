#!/usr/bin/env python3
"""Render a self-contained multi-point SID recovery diagnostic."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Callable


WIDTH = 1200
HEIGHT = 720


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PiperX SID recovery vectors")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _projector(
    points: list[list[float]],
    *,
    horizontal: int,
    vertical: int,
    left: float,
    top: float,
    width: float,
    height: float,
) -> Callable[[list[float]], tuple[float, float]]:
    minimum_x = min(point[horizontal] for point in points)
    maximum_x = max(point[horizontal] for point in points)
    minimum_y = min(point[vertical] for point in points)
    maximum_y = max(point[vertical] for point in points)
    span_x = max(maximum_x - minimum_x, 1e-6)
    span_y = max(maximum_y - minimum_y, 1e-6)
    scale = min(width / span_x, height / span_y) * 0.88
    center_x = (minimum_x + maximum_x) * 0.5
    center_y = (minimum_y + maximum_y) * 0.5

    def project(point: list[float]) -> tuple[float, float]:
        return (
            left + width * 0.5 + (point[horizontal] - center_x) * scale,
            top + height * 0.5 - (point[vertical] - center_y) * scale,
        )

    return project


def _path(points: list[list[float]], project: Callable) -> str:
    projected = [project(point) for point in points]
    return " ".join(
        ("M" if index == 0 else "L") + f" {x:.2f} {y:.2f}"
        for index, (x, y) in enumerate(projected)
    )


def _line(
    start: list[float],
    end: list[float],
    project: Callable,
    *,
    color: str,
    marker: str,
    width: float,
    opacity: float = 1.0,
) -> str:
    start_x, start_y = project(start)
    end_x, end_y = project(end)
    return (
        f'<line x1="{start_x:.2f}" y1="{start_y:.2f}" '
        f'x2="{end_x:.2f}" y2="{end_y:.2f}" stroke="{color}" '
        f'stroke-opacity="{opacity}" stroke-width="{width}" '
        f'marker-end="url(#{marker})"/>'
    )


def _circle(point: list[float], project: Callable, *, color: str, radius: int) -> str:
    x, y = project(point)
    return (
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius}" fill="{color}" '
        'stroke="#08131d" stroke-width="1.5"/>'
    )


def _sample_label(
    point: list[float], project: Callable, *, sample_index: int
) -> str:
    x, y = project(point)
    label = f"P{sample_index + 1}"
    box_width = 28 if sample_index >= 9 else 23
    box_x = x + 6
    box_y = y - 20
    return (
        f'<g><rect x="{box_x:.2f}" y="{box_y:.2f}" width="{box_width}" '
        'height="15" rx="3" fill="#08131d" stroke="#ff7f32" '
        'stroke-width="1"/>'
        f'<text x="{box_x + box_width / 2:.2f}" y="{box_y + 11:.2f}" '
        'text-anchor="middle" fill="#f3f8fb" '
        'font-family="ui-monospace,monospace" font-size="10" '
        f'font-weight="700">{label}</text></g>'
    )


def render(payload: dict) -> str:
    if payload.get("schema") != "robo_visualize.sid_recovery_vectors.v2":
        raise ValueError("unexpected SID recovery vectors schema")
    manifold = payload["manifold"]
    samples = sorted(payload["samples"], key=lambda item: item["sample_index"])
    aggregate = payload["aggregate_metrics"]
    if not samples or aggregate["sample_count"] != len(samples):
        raise ValueError("SID recovery sample count is inconsistent")
    full = [
        pose[:3]
        for pose in manifold["full_trajectory_poses_base_link6_wxyz"]
    ]
    approach = [
        pose[:3]
        for pose in manifold["approach_trajectory_poses_base_link6_wxyz"]
    ]
    point_sets = [
        (
            sample["p_pose_base_link6_wxyz"][:3],
            sample["attached_episode_q_pose_base_link6_wxyz"][:3],
            sample["predicted_endpoint_pose_base_link6_wxyz"][:3],
        )
        for sample in samples
    ]
    all_points = [*full, *(point for points in point_sets for point in points)]
    selection = payload["sample_selection"]
    title = "PiperX SID Recovery Field — Expert Episode 3"
    subtitle = (
        f"{len(samples)} unfiltered Omega samples · seed {selection['seed']} · "
        f"translation radius {selection['translation_radius_m']:.2f} m · "
        f"rotation radius {selection['rotation_radius_rad']:.2f} rad · "
        "base_link / link6"
    )

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
        f'viewBox="0 0 {WIDTH} {HEIGHT}">',
        "<defs>",
        '<marker id="predicted-head" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#47f08b"/></marker>',
        '<marker id="reference-head" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#ffd166"/></marker>',
        "</defs>",
        '<rect width="1200" height="720" fill="#08131d"/>',
        '<rect x="34" y="106" width="550" height="470" rx="18" fill="#102330" stroke="#244151"/>',
        '<rect x="616" y="106" width="550" height="470" rx="18" fill="#102330" stroke="#244151"/>',
        f'<text x="36" y="48" fill="#f3f8fb" font-family="system-ui,sans-serif" font-size="28" font-weight="700">{html.escape(title)}</text>',
        f'<text x="36" y="78" fill="#9bb4c3" font-family="system-ui,sans-serif" font-size="15">{html.escape(subtitle)}</text>',
    ]

    panels = (
        ("Top view · X/Y", 0, 1, 54.0, 126.0),
        ("Side view · X/Z", 0, 2, 636.0, 126.0),
    )
    for label, horizontal, vertical, left, top in panels:
        project = _projector(
            all_points,
            horizontal=horizontal,
            vertical=vertical,
            left=left,
            top=top,
            width=510.0,
            height=420.0,
        )
        parts.extend(
            [
                f'<text x="{left + 14:.0f}" y="{top + 20:.0f}" fill="#d8e7ef" font-family="system-ui,sans-serif" font-size="16" font-weight="650">{label}</text>',
                f'<path d="{_path(full, project)}" fill="none" stroke="#4d9fe8" stroke-opacity="0.48" stroke-width="2"/>',
                f'<path d="{_path(approach, project)}" fill="none" stroke="#31e3d1" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>',
            ]
        )
        for sample, (p_point, q_point, endpoint) in zip(
            samples, point_sets, strict=True
        ):
            parts.extend(
                [
                    _line(
                        p_point,
                        q_point,
                        project,
                        color="#ffd166",
                        marker="reference-head",
                        width=1.5,
                        opacity=0.62,
                    ),
                    _line(
                        p_point,
                        endpoint,
                        project,
                        color="#47f08b",
                        marker="predicted-head",
                        width=2.5,
                    ),
                    _circle(p_point, project, color="#ff7f32", radius=5),
                    _circle(q_point, project, color="#31e3d1", radius=4),
                    _circle(endpoint, project, color="#47f08b", radius=4),
                    _sample_label(
                        p_point,
                        project,
                        sample_index=sample["sample_index"],
                    ),
                ]
            )

    mean_reduction = 100.0 * aggregate[
        "mean_nearest_manifold_distance_reduction_fraction"
    ]
    mean_cosine = aggregate["mean_base_translation_direction_cosine"]
    parts.extend(
        [
            '<g font-family="system-ui,sans-serif" font-size="14">',
            '<circle cx="52" cy="620" r="6" fill="#ff7f32"/><text x="66" y="625" fill="#d8e7ef">perturbation P1–P10</text>',
            '<circle cx="235" cy="620" r="6" fill="#47f08b"/><text x="249" y="625" fill="#d8e7ef">predicted endpoint / vector</text>',
            '<circle cx="490" cy="620" r="6" fill="#31e3d1"/><text x="504" y="625" fill="#d8e7ef">nearest expert Q</text>',
            '<line x1="680" y1="620" x2="720" y2="620" stroke="#ffd166" stroke-width="2" stroke-opacity="0.62"/><text x="731" y="625" fill="#d8e7ef">analytic P→Q</text>',
            '<line x1="895" y1="620" x2="935" y2="620" stroke="#31e3d1" stroke-width="4"/><text x="946" y="625" fill="#d8e7ef">approach manifold</text>',
            "</g>",
            f'<text x="52" y="676" fill="#f3f8fb" font-family="ui-monospace,monospace" font-size="16">mean direction cosine {mean_cosine:.4f}   distance reduced {aggregate["distance_reduced_count"]}/{len(samples)}   mean reduction {mean_reduction:.1f}%</text>',
            '<text x="1148" y="688" text-anchor="end" fill="#668494" font-family="system-ui,sans-serif" font-size="12">RoboVisualize · SID link6 multi-point diagnostic</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.resolve().read_text())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(payload), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
