# Dataset processing

## Input contract

The automatic exporter targets LeRobot v3 and PiperX's 14-value state/action:
six joint angles in degrees and one gripper value in millimetres per arm,
left followed by right. Other robots, units, feature layouts, cameras or task
semantics require an explicit adapter. The `automatic_dataset` entry point is
hard-coded to the drawer task; it is not a generic task selector.

Keep raw inputs immutable. Select output and work directories outside the source.
Preserve episode identity, source FPS, camera keys, raw offsets and source hashes.
The main composite combines two source times and has no single captured timestamp.
Each wrist view and its arm's telemetry must follow that arm's source clock.

## Trim

```bash
real-robot-trim --dataset /path/to/raw --output /path/to/trimmed \
  --repo-id owner/trimmed --tail-seconds 2
```

Trimming retains available terminal stillness and does not invent missing frames.
Retiming separately appends a labelled synthetic terminal hold (60 frames at
30 FPS in the automatic drawer exporter). Keep this distinction in provenance.

## Export the drawer dataset

Install PiperX dependencies with `python -m pip install -e '.[piperx]'` and run
`robo-visualize-assets --output /path/to/robot/assets` (see [asset setup](robot-assets.md)).
The geometry integration uses the bundled PiperX model code and verified mesh assets; consult
[third-party dependencies](../THIRD_PARTY.md) for their availability and identity.
The URDF and actual inter-arm base placement must match your recording. In this
source snapshot `PiperXClearance` defaults to 0.59 m base spacing; do not infer a
calibration from historical asset notes mentioning 0.49 m.

```bash
python -m real_robot_data_retime.automatic_dataset \
  --source /path/to/trimmed --raw-source /path/to/raw \
  --output /path/to/retimed --work-dir /path/to/analysis \
  --urdf /path/to/calibrated.urdf --mesh-root /path/to/robot/assets \
  --repo-id owner/retimed
```

Use `--episodes 0 1` to process selected indices without global finalization.
The full invocation finalizes after all source episodes have been processed.
No upload is performed by this command.

## Verify before use

The automatic drawer validator is an API, not the general numeric retiming CLI:

```python
from pathlib import Path
from real_robot_data_retime.automatic_validation import validate
report = validate(Path('/path/to/trimmed'), Path('/path/to/retimed'))
print(report)
```

Check every row and camera stream: source endpoints/order, allowed holds, per-arm
indices, wrist frames, FPS/frame count, task dependencies, object ownership and
contact ledgers. Inspect rendered videos. A nonempty preserved-contact ledger
means original contact edges were retained; do not describe that as absolute
collision freedom. Video-only projected checks cannot replace mesh checks.

IdleMask is task-specific supervision, not a synonym for a repeated video frame.
Retain required waits, contact and coupled intervals; exclude only semantically
qualified idle periods. Verify `[left, right]` ordering and numeric/video alignment.

## Statistics

`stats.compute_dataset_numeric_stats` computes exact per-feature global quantiles
by concatenating dataset rows (`numpy.quantile`, linear interpolation by default).
`stats.rewrite_dataset_numeric_stats` writes numeric statistics for explicitly
selected features. Audit the training split and valid-row/hold/mask policy before
calling it: this utility does not automatically choose a split or remove synthetic
holds. Never average episode quantiles. Bind the consumed statistics to dataset
revision, selected rows, feature order/units, preprocessing and SHA-256. A dataset
metadata file is not proof that a trainer consumed those statistics.
