# CTR: real-robot video and dataset retiming

[中文说明](README_zh.md)

Convert sequential dual-arm demonstrations into temporally overlapping demonstrations.
CTR discovers interactions, schedules per-arm source clocks, composites the main
camera, and remaps each arm's recorded telemetry and wrist video to its own clock.

This first code release covers real-robot data processing. It does not include
policy training or RoboTwin evaluation. The Python package retains its name
`real_robot_data_retime` so the original processing modules remain directly usable.

## What is supported

| Path | Inputs | Output and scope |
| --- | --- | --- |
| Video editor | Fixed-view RGB video; optional task selection | Composite video, interaction evidence and projected foreground checks; built-in drawer, letters and workpiece profiles |
| Automatic dataset exporter | PiperX LeRobot v3 dataset, original raw dataset, URDF and robot meshes | Drawer-specific automatic exporter with synchronized main/wrist videos, action/state and source-clock receipts |
| Task-specific dataset modules | Recorded telemetry and explicit phase/configuration evidence | Workpiece, screw and water retiming utilities; inspect their APIs and task-specific assumptions before use |
| New task with an Agent | Language description, representative video, and dataset/robot specification when available | Agent-written task adapter, processing script and verification report; follow the [new-task workflow](docs/new-task-agent.md) |

A new task is a code adaptation workflow. There is no universal language-to-script
CLI, and selecting a profile does not establish that an unseen scene is supported.

## Installation

Use Python 3.11 and install `ffmpeg` on the host. For GPU segmentation, the source
runtime uses PyTorch 2.8.0, torchvision 0.23.0 and transformers 5.16.1.

```bash
git clone https://github.com/MINT-SJTU/CTR.git
cd CTR
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[interaction,segmentation,test]'
```

The default backend is SAM2. Model identities are pinned in
`src/real_robot_data_retime/segmentation/model_versions.py`. Weights download
separately and retain their upstream terms. Mesh scheduling uses the integrated `robo_visualize` package. Install `.[piperx]`
and run the verified asset downloader described in [robot assets](docs/robot-assets.md).
SO-101 and RoboTwin viewers are also included and use separate environment extras.

## Process a video

```bash
python main.py --input /path/to/source.mp4 --task drawer \
  --output /path/to/parallel.mp4 --debug-dir /path/to/analysis
python main.py --input /path/to/source.mp4 --task drawer \
  --analysis-only --debug-dir /path/to/analysis
python batch.py --input-dir /path/to/videos --output-dir /path/to/outputs
```

Use `letters` or `workpiece` for the other built-in profiles. Reports include
interaction hypotheses, segmentation/tracking evidence, source clocks and
rendering checks. Review the output for arm completeness, duplicated objects,
release ownership, occlusions and transitions. Uncertain evidence must remain
visible in the report rather than being promoted to a successful demonstration.

## Visualize and qualify robot geometry

```bash
python -m pip install -e '.[piperx]'
robo-visualize-assets --output /path/to/piperx-assets
export CTR_PIPERX_ASSETS=/path/to/piperx-assets
robo-visualize-piperx-web --dataset-path /path/to/lerobot-dataset
```

See [asset setup, public sources and viewer environments](docs/robot-assets.md).

## Process a dataset

Start with [dataset processing and validation](docs/datasets.md), including the
PiperX representation, source provenance, camera alignment and normalization rules.
To adapt to a new task, give your coding Agent the [task brief](examples/task-brief.md)
and [Agent workflow](docs/new-task-agent.md). A reusable [starting prompt](examples/agent-prompt.md)
is included.

## Development and verification

```bash
python -m pip install -e '.[interaction,visualization,test]'
python -m pytest -q
```

Some tests need robot geometry or optional model dependencies and are skipped when
those are absent. CPU tests do not certify segmentation quality on your video or
physical robot execution. See [release provenance](docs/release.md) for the exact
source snapshot, packaging scope and verification limitations.

## License

CTR-owned code is released under [MIT](LICENSE). External libraries, models and
robot geometry retain their own terms; see [third-party dependencies](THIRD_PARTY.md).
