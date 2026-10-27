# External dependencies and asset provenance

The MIT license at repository root covers CTR-owned code. It does not relicense
model weights, upstream libraries or robot descriptions/meshes.

- Python runtime dependencies are declared in `pyproject.toml`; install them from
  their upstream distributions and retain their notices.
- SAM2 models are downloaded from `facebook/sam2.1-hiera-small` or
  `facebook/sam2.1-hiera-large`. CoTracker is an optional tracking dependency from
  https://github.com/facebookresearch/co-tracker at the revision in `pyproject.toml`.
  The model revisions in `segmentation/model_versions.py` identify weights, not
  permission to redistribute them. No model weights are bundled.
- Experimental RobotSeg/TAPIR adapters remain separate from the default SAM2
  pipeline. Their runtime/model dependencies are not installed by the base package.
- `collision/piperx.py` currently imports `PiperXModel` from the separate
  `robo_visualize` package. Its source repository is private. This is a release
  blocker for publicly reproducible mesh-based dataset processing until the
  required dependency is made distributable or replaced explicitly.
- The existing `assets/piper_x_description.urdf` is a source snapshot from that
  external project with a gripper mount correction. Its provenance is retained in
  `assets/provenance.json`; redistribution attribution and a public geometry source
  must be resolved before the release is published. Meshes are not bundled.

Do not substitute a similar robot mesh or URDF without verifying joint order,
units, gripper semantics, mounting transforms, base placement and geometry hashes.
