# CTR：真机视频与数据集重定时

[English](README.md)

将双臂顺序演示重组为可重叠执行的演示：识别交互、规划左右臂各自的源时间轴、
合成主视角，并同步重映射 action/state 与腕部视频。本次发布只包含真机数据处理。

## 安装

需要 Python 3.11 和系统 `ffmpeg`。分割使用 GPU PyTorch。

```bash
git clone https://github.com/MINT-SJTU/CTR.git
cd CTR
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[interaction,segmentation,test]'
```

## 视频处理

```bash
python main.py --input /path/to/input.mp4 --task drawer \
  --output /path/to/output.mp4 --debug-dir /path/to/analysis
```

内置视频任务为 `drawer`、`letters`、`workpiece`。
`--analysis-only` 只分析；`batch.py --input-dir ... --output-dir ...` 批处理。
视频路径检查的是投影轮廓，不能替代基于关节和网格的三维检查。

## 数据集处理

自动导出入口针对 PiperX、LeRobot v3 和抽屉任务。左右臂分别采用六个角度值
（度）和一个夹爪值（毫米）；其他机器人或任务必须适配，不能直接套用。
参见[数据集命令、同步和验证](docs/datasets.md)。三维检查还需要机器人运动学、
网格和标定信息，可视化和运动学代码已合入本仓库。安装与资产获取见[机器人资产说明](docs/robot-assets.md)，
公开来源和许可见 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 新任务：语言描述＋视频 → Agent 生成脚本

参见[中文流程说明](docs/new-task-agent_zh.md)、[详细代码入口](docs/new-task-agent.md)、
[任务输入模板](examples/task-brief.md)和[Agent 起始提示词](examples/agent-prompt.md)。
Agent 需要先观察视频并明确任务依赖，再复用现有模块生成适配器与处理脚本，
最后运行测试、处理真实片段并检查画面。这里发布的是可复用的开发流程，
没有宣称提供任意新任务的一键自动生成器。

## 验证与许可

```bash
python -m pip install -e '.[interaction,visualization,test]'
python -m pytest -q -rs
```

CPU 测试不能代替真实视频、数据集和机器人几何验证。
[发布记录](docs/release.md)列出来源、验证结果和限制。
自有代码采用 [MIT](LICENSE)，外部模型、依赖和机器人资产保留各自许可。
