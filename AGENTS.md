# CTR real-robot processing

Read README.md before editing, and docs/new-task-agent.md when adding a task.
Reuse the shared package rather than copying a task pipeline. New profiles need
explicit discovery, scheduling, compositing and export integration; the automatic
dataset exporter is drawer-specific.

Preserve immutable source data and per-arm temporal provenance. Keep all videos,
datasets, model weights, logs and generated artifacts outside Git. Do not swallow
failures or certify unsupported tasks. Distinguish projected checks, mesh checks
and physical execution. Never average local quantiles to approximate global ones.

Run focused tests and the regression suite for code changes. Report skipped
integration checks and real-video validation separately. Do not upload data or
operate a robot unless explicitly requested by the user.
