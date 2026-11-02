# Release preparation record

Prepared on 2026-10-28 for `MINT-SJTU/CTR` from the `main` branch of
`Shiki42/real-robot-data-retime`, snapshot `623770e`.

Included: shared Python processing package, video/batch entry points, focused
processing/review scripts, small task configurations, regression tests, browser
review assets, installation instructions, new-task Agent workflow and MIT license.
The source history is not imported. Existing publication-repository history is
preserved; its old tracked paper files are removed by a normal descendant commit.

Excluded: historical experiment directories/receipts, archived frozen drawer
pipeline, paper files, private outputs, videos, datasets, weights, automatic upload
scripts and per-episode repair/retry orchestration. Three small screw configurations
needed by regression tests have moved to `tests/fixtures/`.

Verification on the preparation host:

- Existing processing environment: 358 tests passed; 4 skipped.
- Fresh Python 3.11 environment installed from `.[interaction,test]`:
  354 tests passed; 8 skipped (optional torch/hppfcl and real-data integration).
- Video and automatic-dataset CLIs import and expose `--help`.
- Wheel builds and includes both HTML preview resources.
- Browser mask-timeline and preview JavaScript syntax checks passed.
- Pre-commit review covered code reuse, task dispatch, dependencies, packaging,
  local documentation links, accidental large payloads and credential patterns.

These are CPU/package checks. No new real-video GPU segmentation or complete
LeRobot export has been certified for this release. Tests requiring a generated
drawer episode or real scene remain separate acceptance checks. No policy training,
evaluation or robot motion was launched.

The private geometry dependency was resolved by integrating RoboVisualize commit
`cb5cca927b9be76a5a1b8a953a061a233ed3ec1f`. PiperX assets were downloaded from
fixed public revisions and all 12 output files were SHA-256 verified. The
integrated suite passed 372 tests with 4 skips, including PiperX kinematics and
real mesh collision checks. The remaining skips concern real-video/dataset
fixtures and optional torch probes. This is not a newly certified video
segmentation or full dataset-export result. SO-101/RoboTwin source adapters are
included; their complete optional environments were not qualified in this check.

See [THIRD_PARTY.md](../THIRD_PARTY.md) and [robot assets](robot-assets.md) for
source hashes, download/bundling policy, upstream notices and geometry limitations.

The merged package also passed a clean CPU-only install (364 passed, 12 optional
integration skips), wheel/sdist resource and license checks, an installed-wheel
asset download with all 12 hashes verified, and wheel-only PiperX model loading
with 10 visual meshes. Browser mask-timeline and preview syntax checks passed.

Final source-environment regression with the merged package and verified PiperX
assets: **374 passed, 2 skipped**. Only the real-video scene fixture and generated
drawer-dataset fixture were unavailable. No new GPU video processing or complete
dataset export was run as part of packaging.


## Raw-episode verification update (2026-11-02)

The original packaging checks above describe the 2026-10-28 snapshot. The later
four-task raw-episode check runs on lab machine b, using one pinned episode per
task and the published package as its base. It adds the portable raw runner and
reuses the complete water implementation from source commit
`6733992ec0eda1cc67d385f553501626bb248dd6`.

See [raw-episode processing](raw-episode-processing.md) for commands and
[the four-task verification record](raw-verification.md) for actual inputs,
outputs and the limits of acceptance. Detailed logs, failed attempts, segmentation
caches, source videos and generated outputs remain outside Git. This processing
does not launch policy training, evaluation or robot motion.
