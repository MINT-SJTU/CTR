# From a task description and video to a processing script

This workflow is intended for a coding Agent that can inspect video frames, read
this repository, edit Python and execute validation. It does not require a specific
Agent vendor. The repository does not ship a language model orchestrator. Video-only
input can produce edited RGB demonstrations; synchronized robot training data also
requires recorded state/action and camera streams. Do not infer joints from pixels.

## 1. Freeze the task contract

Ask the user for the language goal and representative video. Use the
[task brief](../examples/task-brief.md) to record object identities, arm assignment,
start/end conditions, permitted overlap and required dependencies. For dataset
output, obtain the dataset schema, camera mapping, units, robot geometry and
calibration. If the goal is ambiguous (for example holding versus releasing an
object), resolve it before encoding task logic.

## 2. Inspect evidence before choosing an implementation

Measure video FPS, dimensions and frame count; inspect frames throughout the
sequence, including occlusion and contact transitions. Create an event table with
frame/time intervals, arm, object, approach/grasp/transport/release/retreat,
visibility and uncertainty. Mark source observations separately from inferred
intervals. A user description supplies intent; images supply evidence of what
actually occurred. Neither should silently overwrite the other.

Inspect the existing implementation before adding code:

| Responsibility | Existing code |
| --- | --- |
| CLI and batch entry | `main.py`, `batch.py` |
| Interaction inference | `interaction/pipeline.py`, `interaction/discovery.py`, `interaction/evidence.py`, `interaction/verification.py` |
| Task priors | `tasks/__init__.py`, `tasks/drawer.py`, `tasks/letters.py`, `tasks/workpiece.py` |
| Masks and tracking | `segmentation/`, `tracking/` |
| Time maps and waits | `timeline/`, including `scheduler.py` and `smooth.py` |
| Rendering and ownership | `edit.py`, `compositing/`, `background/` |
| Joint clearance | `collision/piperx.py` |
| Dataset remapping | `automatic_dataset.py`, `dataset.py`, task-specific modules |
| Validation and statistics | `automatic_validation.py`, `validate.py`, `stats.py` |

All package-relative paths above are below `src/real_robot_data_retime/`.

## 3. Generate a small task adapter and explicit runner

Choose an existing task only when its semantics match. A new profile dictionary
alone is insufficient: inspect task dispatch and hard-coded branches in discovery,
interaction, planning, rendering and export. Add a focused task adapter and wire
those branches explicitly, including CLI task choices. Reuse the existing source
clock, interpolation, compositor and validator; do not copy an entire pipeline.
Do not disguise a new task as `drawer` merely to pass through the exporter.

The Agent should produce:

- A task specification containing observed events and required precedence edges.
- A task adapter with object/arm ownership and task-specific scheduling constraints.
- A runnable script accepting explicit source, output and work paths, with no
  user-specific paths, embedded credentials or automatic upload.
- Validation checks and a report relating every output interval to source evidence.

The generated runner should separate analysis, planning, rendering and validation
so expensive measurements can be reused only when source/config/model identity
matches. Expose uncertain events for review and fail with their evidence when
required events cannot be established. Do not fabricate confident boundaries.

## 4. Plan monotone per-arm source clocks

Preserve each arm's manipulation order and required contact intervals. Encode
cross-arm dependencies such as opening before insertion and withdrawing before
closing. Choose semantically valid waiting poses; verify held objects remain held.
Use the shared smoothing implementation for braking/restart. Check the retimed
path after smoothing, including swept transitions, rather than only source poses.

For video-only output, report projected mask clearance and its occlusion assumptions.
For telemetry output, use calibrated robot geometry and record mesh/scene margins,
held-object assumptions and every retained source-contact exception. Never replace
an unavailable geometry check with a passing two-dimensional check.

## 5. Export aligned observations and actions

Use each arm's source clock for its joints, action, wrist camera and arm-specific
telemetry. Composite the main camera using explicit foreground/background and
object ownership. Verify source-origin removal after pickup, shared-scene ownership
after release, occlusion and clean-plate availability. Preserve raw source IDs,
indices, fractional interpolation, synthetic holds and calibration/model identities.
Do not alter frame rate or mask semantics merely to simplify export.

## 6. Acceptance criteria

Run focused tests for the new task and the existing regression suite. Then process
a representative real episode and inspect the rendered video; unit tests alone
cannot certify a new scene. Include failure examples involving occlusion, ambiguous
release, dependency violations and unsafe clock combinations.

Accept only when the intended goal and precedence hold, no manipulation disappears,
source clocks are monotone, cameras/telemetry agree, numerical rows are finite,
object ownership remains consistent and checks cover every generated transition.
Report durations, evidence paths, unresolved uncertainty and the precise geometry
scope. Freeze the adapter/config/source/model hashes before scaling to a batch.
