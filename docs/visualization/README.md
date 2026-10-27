# Integrated RoboVisualize reference

RoboVisualize source is included under `src/robo_visualize`. Installation, verified
PiperX asset download and separate environment requirements are documented in
[robot-assets.md](../robot-assets.md). No second repository checkout is needed.

## Modules and entry points

- `web/`: shared Flask backend and browser viewer.
- `arms/piperx/`: PiperX LeRobot dataset playback, forward kinematics and a live
  state-file viewer. Run `robo-visualize-piperx-web --help` or
  `robo-visualize-piperx-live-web --help` for their explicit input paths.
- `arms/so101/`: dual SO-101 dataset viewer and Open3D renderer, exposed as
  `robo-visualize-so101-web` and `robo-visualize-so101-render`.
- `arms/robotwin/`: dual-arm RoboTwin HDF5/action viewer, exposed as
  `robo-visualize-robotwin-web`. Its robot assets remain external.

PiperX playback consumes six joint angles in degrees and a gripper value in
millimetres. Viewer playback, mesh clearance and live state-file visualization
are distinct operations; none sends control commands to a robot.

## Optional recovery overlays

PiperX supports explicit perturbation-point and recovery-vector sidecar files.
The recovery overlay's v2 schema is multi-point and uses `base_link` coordinates
and `link6` as the controlled frame. It displays expert trajectories, predicted
endpoints and reference recovery directions. See
`tests/robo_visualize/test_sid_recovery.py` for a synthetic schema example and
`scripts/render_sid_recovery_svg.py --help` for SVG generation.

The live viewer reads already-generated state/trajectory files. Its tests use
synthetic state records in `tests/robo_visualize/test_live_server.py`; no policy
checkpoint, experiment result or demonstration dataset is distributed.

## Extend the viewer

Implement the shared `robo_visualize.web.server.Backend` interface in an arm
adapter and reuse `create_app`. Adapters provide configuration, episode payload,
packed transforms, mesh paths and camera-video paths. Specify units, frame names,
base transforms and camera mappings explicitly. Keep datasets and outputs outside
Git and record source identities in any derived artifacts.
