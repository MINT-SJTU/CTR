# Dataset review components

`idle-timeline.js`, `idle-timeline.css` and `multi-camera.js` are shared by the
screw/workpiece players. Players read `observation.arm_active_mask` from the same
per-frame values stored in Parquet: `[left, right]`, 1 supervised, 0 excluded.
Orange denotes proposed loss exclusion, not a general physical-stillness detector.
Required waits/contact intervals need task-specific supervision rules.

The included `scripts/refresh_workpiece_review.py` refreshes an existing generated
workpiece preview from its Parquet data. These assets alone do not create a
review dataset or provide segmentation. Generated videos/timelines remain outside
Git. Inspect source clocks, masks, camera alignment and object ownership together.

Run `node tests/web/arm-mask-timeline.cjs` for the timeline/UI regression checks.
