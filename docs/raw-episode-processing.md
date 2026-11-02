# Process one raw episode

The portable runner is `scripts/process_raw_episode.py`. It accepts the four
real-robot tasks `drawer`, `letters`, `workpiece`, and `water`. It calls the shared
trimmer, interaction discovery, SAM2 segmentation, task scheduler and compositor;
it does not train a policy or send robot commands.

## Inputs

Provide an immutable LeRobot v3 raw dataset with the selected episode's data
Parquet, episode metadata, `meta/info.json`, `meta/tasks.parquet`, and RGB videos.
State/action must use the supported left-then-right 14-value PiperX layout.
Record the exact dataset repository and revision. When downloading a subset,
verify every consumed file against that revision's Git blob or LFS SHA-256.
Do not describe a partial download as a complete snapshot.

Drawer and letter tasks use `observation.images.top`; workpiece and water use
`observation.images.right_environment_1`. Wrist views retain each arm's clock.
Source metadata is never rewritten by this runner. Keep output outside the source.

## Prepare on CPU

```bash
python scripts/process_raw_episode.py prepare \
  --task drawer --source /path/to/raw --episode 0 \
  --source-repo owner/raw-dataset --source-revision FULL_HF_COMMIT \
  --output /path/to/new-output
```

This verifies finite N x 14 state/action, detects task-specific trimming, writes
a main-view trimmed video, and checks its complete decoded frame count and FPS.
The output directory must be new. Zero common leading stillness correctly removes
zero frames. Water detects bottle/tray boundaries from the frozen action/state
and records its phase evidence in `detected-phases.json`.

## Segment, reorder and render

```bash
python scripts/process_raw_episode.py render \
  --task drawer --source /path/to/raw --episode 0 \
  --source-repo owner/raw-dataset --source-revision FULL_HF_COMMIT \
  --output /path/to/new-output --gpu-memory-gib 3
```

Install the segmentation dependencies from README first. The memory option is a
PyTorch allocator limit, not a reservation of an otherwise busy GPU. Resource
ownership/sharing must be resolved before launching. For drawer point tracking
on a constrained GPU, pass `--point-tracker-device cpu`; this runs the same
CoTracker model on CPU without reducing its frames, resolution or precision. Process episodes sequentially
when sharing one GPU.

The render stage rejects changed input hashes. For drawer/letters/workpiece it
runs the existing video editor, saves per-arm source clocks, remaps the original
14-D action/state, exports the two wrist videos, and checks numeric interpolation,
clock bounds/monotonicity, and all three decoded video lengths/FPS. Failed
interaction or object-origin gates raise an error rather than certifying a clip.
These are projected video checks, not calibrated three-dimensional robot clearance.

Water runs the dedicated `real_robot_data_retime.water` entry with one
sample selected by `--water-first-arm` and `--water-delay-ratio` (defaults: left,
0.5). The verified episode-0 example uses left and 0.75; other timings can be
rejected by foreground-clearance checks. It keeps tray carrying synchronized and writes the existing LeRobot
export and `VALIDATION.json`. Its origin-audit and visual-review fields remain
visible; numeric export alone does not certify the composite's appearance.

## Outputs and review

`prepared.json` binds the exact source and trim. The first three tasks produce
`parallel.mp4`, `left_wrist.mp4`, `right_wrist.mp4`, `trajectories.parquet`, and
`analysis/` with interaction, segmentation and source-clock evidence. Their
trajectory Parquet is a validation artifact, not an independently finalized
training dataset. Water exports under `water/dataset/` with its own provenance.

`result.json` records actual duration, memory cap and check results after a
completed render. Inspect task events, arm completeness, object origin/release
ownership and synchronization in the resulting video before visual acceptance.
Keep videos, raw subsets, weights, logs and detailed receipts outside Git.
