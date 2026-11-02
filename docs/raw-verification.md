# Four-task raw-episode verification

On 2026-11-02 one pinned raw episode per task was processed on lab machine b.
No policy training, policy evaluation or robot actuation was launched. All views
are 30 FPS. This verifies the release package plus the fixes in this update.

| Task | Episode | Raw frames | Trimmed | Output | Both clocks advancing |
| --- | ---: | ---: | ---: | ---: | ---: |
| drawer | 0 | 634 | 634 | 670 | 356 |
| letters | 0 | 1084 | 1083 | 1086 | 339 |
| workpiece | 0 | 1092 | 1092 | 656 | 361 |
| water | 0 | 736 | 724 | 1024 | 457 |

The last column counts output steps where both arm source clocks advance.
Output length includes required waits, smoothing and terminal holds; retiming
does not guarantee a shorter video.

## Frozen raw inputs

- `drawer`: [Travor278/piperx-put-cube-in-drawer-20260908-87ep](https://huggingface.co/datasets/Travor278/piperx-put-cube-in-drawer-20260908-87ep/tree/58bbbd720f6e78d162b8f4bc7077759d34c5162f), revision `58bbbd720f6e78d162b8f4bc7077759d34c5162f`.
- `letters`: [Shiki42/piperx-sortletter-0905-53ep-raw](https://huggingface.co/datasets/Shiki42/piperx-sortletter-0905-53ep-raw/tree/3dcbd2f1795de8f0549a369e19e168127991b82b), revision `3dcbd2f1795de8f0549a369e19e168127991b82b`.
- `workpiece`: [Shiki42/piperx-workpiece-storage-0909-62ep-raw](https://huggingface.co/datasets/Shiki42/piperx-workpiece-storage-0909-62ep-raw/tree/bd2d9271550e003d2826f11cfb9d95d4b71c23b7), revision `bd2d9271550e003d2826f11cfb9d95d4b71c23b7`.
- `water`: [Takizz/piperx-water-delivery-0920-0921-75ep-raw](https://huggingface.co/datasets/Takizz/piperx-water-delivery-0920-0921-75ep-raw/tree/64a90c0267db28f66fa22588e41ab5990c3e4e0b), revision `64a90c0267db28f66fa22588e41ab5990c3e4e0b`.

Only selected episode files and required metadata were downloaded. Every consumed
file was verified against its pinned Git blob or LFS SHA-256; these are not complete
dataset snapshots. Raw task strings are preserved. Water metadata describes holding
two bottles, while the video also shows tray placement/carrying; no silent relabeling.

## Reproduce

Use the [raw runner](raw-episode-processing.md): `prepare` then `render` with the
same source repository, revision, episode and output. For drawer, add
`--point-tracker-device cpu`; the same CoTracker model, FP32 precision, resolution,
frame stride and temporal window are retained. Feature extraction uses eight-frame
batches. For water, add `--water-first-arm left --water-delay-ratio 0.75`.

The shared GPU PyTorch allocator cap was 3 GiB. Environment: Python 3.11.14,
PyTorch 2.8.0+cu128, torchvision 0.23.0, transformers 5.16.1, numpy 1.26.4.
Model revisions are pinned in `segmentation/model_versions.py`. Bind `HF_HOME`
to the downloaded model cache after reconnecting; use `HF_HUB_OFFLINE=1` once
all weights are available.

## Findings and fixes

- Drawer: an occluded cube track latched onto a colored robot part. Bounded recovery
  reseeds from a newly observed stationary deposition inside the detected drawer,
  excluding static colors and robot-attached distractors. Recovered release is
  source frame 382, before closure at 435. Without qualified joint compaction,
  retain all native right-arm poses so smoothing cannot cross an unchecked gap.
- Letters: interaction and origin gates passed for the selected manipulation spans.
- Workpiece: four transfers were identified; output pickup order is L1, R1, L2, R2.
- Water: SAM included a static desk rail in an arm mask. Exclude only pixels matching
  an observed initial background within bounded appearance/camera uncertainty,
  retaining contrasting foreground. Left-first p=0.75 passed zero-overlap checks
  for both independent phases; p=0.5 stayed rejected. Paired tray carrying retains
  equal native source clocks for frames 263 through 625.

Failed attempts and diagnoses remain in external receipts. Corrected segmentation
and point tracks were reused after revalidation; successful retry durations are
not cold end-to-end timings. The water implementation is reused from source commit
`6733992ec0eda1cc67d385f553501626bb248dd6`.

## Acceptance and limits

All main/wrist videos fully decode at the expected lengths/FPS. Numeric source-clock
and action/state checks passed, as did interaction and object-origin checks. Six
integer-source wrist samples per task matched raw RGB observations within MAE
8/255. Eight main-view frames per task were inspected. This is sampled visual
acceptance, not exhaustive frame-by-frame or calibrated 3-D collision certification.

Drawer, letters and workpiece produce selected manipulation videos and aligned
telemetry traces. They do not establish full native approach/home-pose preservation
or finalized training datasets. Water retains `training_ready=false` and its machine
`visual_review=pending`; the separate receipt records sampled visual inspection.

The final CPU suite passed **371 tests, with 8 optional integration skips**.
Videos, weights, raw subsets, detailed hashes, retry logs and receipts stay outside
Git. On b they are grouped under
`/mnt/b443bebc-54be-454c-80c2-2e202e2a9883/zhaobo/ctr-release-validation-20261102/`.
Entry points are `verified-results.json`, `runs/<task>/result.json`, and
`runs/<task>/independent-qa.json`.
