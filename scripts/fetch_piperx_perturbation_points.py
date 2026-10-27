#!/usr/bin/env python3
"""Stream the PiperX AB RGB-D TAR export and write only P poses to JSONL.

The HF dataset is packaged as one TAR split into eight fixed parts. This script
downloads each remote part lazily and writes augmentation.p_pose_base_f64 rows
without extracting images, producing a small JSONL sidecar for robo-visualize.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterator

import pyarrow.parquet as pq

PARTS = tuple(
    f"https://huggingface.co/datasets/Travor278/piperx-old-ab-rgbd-5903/resolve/main/piperx_old_ab_5903_rgbd_complete_20260826.{index:02d}.part"
    for index in range(8)
)
PREFIX = "formal_10000_rgbd_v6_sharded_live_a_v1/"


class ChainedReader(BinaryIO):
    def __init__(self, streams: Iterator[BinaryIO]) -> None:
        self._next_stream = streams
        self._current: BinaryIO | None = None

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            chunks = []
            while True:
                chunk = self.read(1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            return b"".join(chunks)
        while self._current is not None or True:
            if self._current is None:
                try:
                    self._current = next(self._next_stream)
                except StopIteration:
                    return b""
            chunk = self._current.read(size)
            if chunk:
                return chunk
            self._current.close()
            self._current = None
        return b""


def open_remote(url: str) -> BinaryIO:
    request = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    return urllib.request.urlopen(request, timeout=60)


def is_data_parquet(path: str) -> bool:
    return path.endswith(".parquet") and "/data/chunk-" in path


def extract_points(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        with contextlib.closing(ChainedReader(iter(open_remote(url) for url in PARTS))) as reader:
            with tarfile.open(fileobj=reader, mode="r|") as tar:
                for member in tar:
                    if not is_data_parquet(member.name):
                        continue
                    stream = tar.extractfile(member)
                    if stream is None:
                        continue
                    with stream:
                        table = pq.read_table(
                            io.BytesIO(stream.read()),
                            columns=[
                                "augmentation.p_pose_base_f64",
                                "augmentation.q_pose_base_f64",
                                "augmentation.point_id",
                                "augmentation.projection_residual_m_f64",
                            ],
                        )
                    for row in table.to_pylist():
                        pose = row["augmentation.p_pose_base_f64"]
                        handle.write(
                            json.dumps(
                                {
                                    "point_id": row["augmentation.point_id"],
                                    "augmentation.p_pose_base_f64": [float(v) for v in pose],
                                    "augmentation.q_pose_base_f64": [
                                        float(v) for v in row["augmentation.q_pose_base_f64"]
                                    ],
                                    "augmentation.projection_residual_m_f64": (
                                        float(row["augmentation.projection_residual_m_f64"])
                                        if row["augmentation.projection_residual_m_f64"] is not None
                                        else None
                                    ),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        seen += 1
                        if seen % 1000 == 0:
                            print(f"processed {seen} points", flush=True)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("piperx_perturbation_points.jsonl"))
    args = parser.parse_args()
    count = extract_points(args.output)
    print(f"wrote {count} points to {args.output}")


if __name__ == "__main__":
    main()
