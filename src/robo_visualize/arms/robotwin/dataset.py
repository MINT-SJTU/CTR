from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


EPISODE_PATTERN = re.compile(r"episode(?P<index>\d+)\.hdf5$")


@dataclass(frozen=True)
class RoboTwinDataset:
    name: str
    root: Path
    episode_indices: tuple[int, ...]

    @classmethod
    def load(cls, name: str, root: str | Path) -> "RoboTwinDataset":
        path = Path(root).resolve()
        data_dir = path / "data"
        if not data_dir.is_dir():
            raise FileNotFoundError(f"RoboTwin data directory missing: {data_dir}")
        indices = []
        for hdf5_path in data_dir.glob("episode*.hdf5"):
            match = EPISODE_PATTERN.fullmatch(hdf5_path.name)
            if match is None:
                continue
            episode_index = int(match.group("index"))
            head_videos = (
                path / "video" / f"episode{episode_index}.mp4",
                path / "videos" / "head" / f"episode{episode_index}.mp4",
            )
            if not any(video.is_file() for video in head_videos):
                raise FileNotFoundError(
                    f"matching RoboTwin head video missing: {head_videos}"
                )
            indices.append(episode_index)
        if not indices:
            raise ValueError(f"RoboTwin dataset has no episodes: {path}")
        return cls(name, path, tuple(sorted(indices)))

    def episode_path(self, episode_index: int) -> Path:
        self.validate_episode(episode_index)
        return self.root / "data" / f"episode{episode_index}.hdf5"

    def recorded_video_path(self, episode_index: int, camera: str) -> Path:
        self.validate_episode(episode_index)
        candidates = [
            self.root / "videos" / camera / f"episode{episode_index}.mp4",
        ]
        if camera == "head":
            candidates.append(self.root / "video" / f"episode{episode_index}.mp4")
        matches = [path for path in candidates if path.is_file()]
        if len(matches) > 1:
            raise ValueError(
                f"multiple recorded {camera} videos for episode {episode_index}: {matches}"
            )
        if matches:
            return matches[0]
        raise FileNotFoundError(
            f"recorded {camera} video missing for episode {episode_index}: {candidates}"
        )

    def validate_episode(self, episode_index: int) -> None:
        if episode_index not in self.episode_indices:
            raise ValueError(f"episode {episode_index} is not present in {self.name}")
