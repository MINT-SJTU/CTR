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

Publication is pending resolution of the private geometry dependency described in
[THIRD_PARTY.md](../THIRD_PARTY.md). The candidate lives on the preparation host;
this document does not claim a published release or public end-to-end reproducibility.
