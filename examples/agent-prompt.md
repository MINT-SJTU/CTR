Read README.md and docs/new-task-agent.md in this checkout. Use the attached task
brief, language goal and source video to generate a runnable processing script
and the smallest necessary task adapter. Inspect video evidence and existing
code before changing anything. Resolve ambiguities in the task goal first.

Reuse the existing interaction, source-clock scheduling, smoothing, compositing,
dataset remapping and validation modules. Identify hard-coded task assumptions;
do not assume that adding a profile enables a new task end to end. Keep source
inputs immutable and use explicit input/output/work paths.

First produce the measured event table, task dependencies and adapter design.
Then implement and test it. Process a representative episode, inspect the output,
and report source provenance, checks, durations and unresolved evidence. For a
dataset, verify each arm's state/action/wrist camera alignment and compute global
statistics over the declared valid training rows. Distinguish projected video
checks from calibrated geometry checks. Do not mark missing evidence as success.

Deliver the script and invocation, adapter/configuration, focused tests, rendered
preview and verification report. Do not upload data or execute robot motions.
