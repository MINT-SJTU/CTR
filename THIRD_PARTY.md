# Third-party code, models and robot assets

CTR-owned processing code and the user-authorized RoboVisualize integration use
[MIT](LICENSE). This does not relicense upstream robot assets or model weights.
No model weights are distributed in this repository.

## RoboVisualize code

The `robo_visualize` package and its supporting scripts were integrated from
`Shiki42/robo-visualize` commit `cb5cca927b9be76a5a1b8a953a061a233ed3ec1f`.
It is part of this distribution; a private repository is no longer required to
import the PiperX kinematics used by the retiming pipeline.

## Robot geometry

Full file hashes and source paths are recorded in [robot-sources.json](assets/robot-sources.json).
The runtime download manifest is `src/robo_visualize/asset-manifest.json`.

| Asset group | Public source and fixed revision | Distribution |
| --- | --- | --- |
| PiperX arm meshes J1–J5 and Link6 | [AgileX piper_isaac_sim](https://github.com/agilexrobotics/piper_isaac_sim/tree/8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2/piper_x_description) | Downloaded directly and SHA-256 verified; upstream has no declared repository license, so these files are excluded from the MIT code distribution |
| Piper gripper meshes | [AgileX piper_ros](https://github.com/agilexrobotics/piper_ros/tree/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description/meshes) | Downloaded and verified; upstream MIT notice retained in `licenses/` |
| PiperX base mesh and no-gripper URDF | [Evo-RL public assets](https://github.com/linsun449/evo-rl/tree/5a63e2ba7002f4c76a289af5ef9e93928aae8779/src/lerobot/assets/piper_x_description) | Bundled snapshots match the public Git LFS SHA-256 OIDs; Apache-2.0 repository notice retained. LFS media was unavailable during preparation, so no runtime dependency on that endpoint remains |
| SO-101 meshes and base URDF | [SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100/tree/a758567c3978dfeefe282ede0500085a48fe8f78/Simulation/SO101) | Bundled with Apache-2.0 notice; all 13 meshes match upstream byte for byte. The source project's adapted URDF is identified separately in provenance |
| RoboTwin embodiment assets | [RoboTwin](https://github.com/RoboTwin-Platform/RoboTwin/tree/ea8b21121ebb3cd201ff5b3fe361944ac94eda3f) | Not bundled; obtain assets using the upstream instructions, provide an explicit URDF or set `ROBOTWIN_ROOT`, and record the actual asset identity |

The PiperX gripper/TCP URDF is a RoboVisualize adaptation of the no-gripper
robot description. CTR also retains a separate LF-normalized URDF with a
90-degree gripper mount correction in `assets/`; these are distinct snapshots,
not interchangeable calibrations. See `assets/provenance.json` and the geometry
setup instructions. Repository-level license notices are provenance, not a claim
that CTR owns third-party geometry. Users should check upstream terms for their use.

## Libraries and pretrained models

Dependencies are declared in `pyproject.toml` and installed separately. SAM2 and
CoTracker weight revisions are recorded in
`src/real_robot_data_retime/segmentation/model_versions.py`; upstream weight terms
continue to apply. Experimental RobotSeg/TAPIR adapters have separate runtime and
weight requirements and are not the default video backend. The viewer loads
Three.js 0.164.1 from jsDelivr; offline deployments must provide those browser
resources explicitly. No datasets, credentials or policy checkpoints are bundled.
