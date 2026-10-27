# Robot assets and visualization

The processing and visualization packages now install from the same checkout.
PiperX kinematics no longer need a private `robo-visualize` installation.

## PiperX

Use a separate Python 3.11 environment for this stack:

```bash
python -m pip install -e '.[interaction,piperx,test]'
robo-visualize-assets --output /path/to/piperx-assets
export CTR_PIPERX_ASSETS=/path/to/piperx-assets
robo-visualize-piperx-web --dataset-path /path/to/lerobot-dataset
```

The asset command downloads nine files from fixed public revisions and copies
three attributed snapshots from this package. Every file is SHA-256 checked,
including an existing cache. A mismatch raises an error without overwriting the
file; remove the corrupt file explicitly before downloading again. The final
`verified-assets.json` is written only after the complete set passes. The default
location is `~/.cache/ctr/piperx`; set `CTR_PIPERX_ASSETS` when using another location.
The command does not download any videos, datasets or model weights.

For the automatic drawer exporter use:

```bash
python -m real_robot_data_retime.automatic_dataset \
  --source /path/to/trimmed --raw-source /path/to/raw \
  --output /path/to/retimed --work-dir /path/to/analysis \
  --urdf assets/piper_x_description.urdf --mesh-root "$CTR_PIPERX_ASSETS" \
  --repo-id owner/retimed
```

This command intentionally uses the existing CTR gripper-mount snapshot. The
PiperX viewer defaults to the RoboVisualize snapshot in the asset cache. Verify
which description, joint semantics and calibration match your recording before
choosing either. The collision checker defaults to 0.59 m base spacing; the
original dual-viewer documentation used 0.49 m. Neither value certifies the base
spacing in a new recording. Other calibrations require explicit configuration.

Enable the mesh integration tests with the downloaded geometry:

```bash
CTR_PIPERX_ASSETS=/path/to/piperx-assets \
ROBOVISUALIZE_ASSETS=/path/to/piperx-assets python -m pytest -q -rs
```

The live viewer reads a state file and optional expert sidecar; it does not
control a robot. See `robo-visualize-piperx-live-web --help` and the source's
[visualization reference](visualization/README.md) for its supported sidecar schema.

## SO-101 and RoboTwin viewers

These adapters use Pinocchio 3.9, whereas PiperX/collision uses Pinocchio 2.7.
Install them in separate virtual environments; do not install conflicting extras
into one environment.

```bash
# SO-101 environment
python -m pip install -e '.[so101]'
robo-visualize-so101-web --datasets-root /path/to/lerobot-datasets

# RoboTwin environment
python -m pip install -e '.[robotwin]'
export ROBOTWIN_ROOT=/path/to/robotwin
robo-visualize-robotwin-web --dataset task=/path/to/dataset --urdf /path/to/robot.urdf
```

SO-101 assets are bundled with upstream notices. RoboTwin assets remain external;
follow its public repository's asset instructions. No viewer assumes the original
author's machine or dataset directory. The full source adapters are included, but
PiperX is the geometry stack qualified with the real-robot retiming release.

## Provenance

See [THIRD_PARTY.md](../THIRD_PARTY.md), [robot-sources.json](../assets/robot-sources.json),
and the runtime manifest for public URLs, fixed revisions, file hashes and the
distinction between exact upstream files and modified descriptions. Keep your
recording's actual URDF/mesh/calibration hashes alongside every derived dataset.
