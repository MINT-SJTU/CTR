# CTR 数据处理流程图

画布：180 × 67.5 mm，适合双栏通栏插图。所有文字、箭头、时间轴、图标、联合进度曲线和示意掩码均为矢量；视频帧嵌入 SVG 和 PDF。

第三版进一步压缩上下留白；移除了任务描述框及其箭头，数据集到代码改为直角虚线箭头。左右两张联合进度图的绘图区边长一致。每个重组样本的四张照片均紧贴排列；第三个样本中 Open drawer 的绿色照片在前，Pick up cube 的橙色照片在后，时间轴也以绿色先行。橙色时间轴仍统一在上、绿色在下。RobotSeg 与 Foreground segmentation 标题同一基线。底部 ReComposed DataSet 已移除。

## 交付文件

- `ctr_pipeline.pdf`：论文插图。
- `ctr_pipeline.svg`：可编辑矢量源文件，分组对应四个模块、代码生成与审查、RobotSeg、CTR。
- `ctr_pipeline_preview.png`：预览。
- `modules/`：可单独复用的模块。
- `assets/frames.json`：原始视频时间点与正方形裁剪范围。
- `figure_metadata.json`：颜色及素材性质说明。
- `build_figure.py`：完整布局与矢量绘图源文件；使用 Python、Pillow 和 ReportLab。
- `assets/joint_progress.py`：用户指定的联合进度绘图函数副本；保留原始几何及 OKLab 插值，在主绘图文件中注入 teaser 配色。

## 配色与参考

样式参考 `/Users/shuyuan/Documents/CTR/Paper/teaser/current/ctr_teaser_v4.svg`。

| 内容 | 颜色 |
| --- | --- |
| Pick up cube | `#CC8068` |
| Open drawer | `#4F9E98` |
| Place cube | `#648BB7` |
| Close drawer | `#977CB0` |
| CTR | `#8870B5` |

CTR 模块沿用 teaser 的淡紫背景、重组图标、Keep local arm motion 文案以及绿色保护锁。

## 必须保留的素材说明

1. 原始视频的真实时间顺序是先打开抽屉，再拾取方块。本图按用户指定的“接近、拾取、握住把手、完全打开、放入、部分关闭”顺序呈现，并在图中明确标为 schematic task order。取帧时间没有更改，也没有生成新的机器人动作。
2. 模块三的掩码是依据照片手工绘制的可编辑轮廓示意，不是真实 RobotSeg 推理结果；按照用户第二轮排版要求，图中的两行解释文字已移除，此素材性质保留在本说明和元数据中。待提供真实掩码后可按相同取帧位置与裁剪范围替换。
3. 三个橙青时间轴样本属于概念示意，不表示三段实测视频。第三版已按用户要求移除底部重组视频缩略图；此前从 `episode_000.mp4` 提取的素材仍保留在 assets 中。
4. 联合进度图使用 13 条示意单调轨迹表示可行时序覆盖，不是经验概率密度估计。
5. 最后两帧属于一个受依赖关系保护的阶段，内部保留“放入在前、关闭在后”；拾取与打开两组在前段可重叠。图中的条长只表示概念时序，不按视频秒数比例绘制。

## 复现

在 macOS 安装 ReportLab 与 Pillow 后运行：

```sh
python3 build_figure.py
pdftoppm -scale-to 2700 -png -singlefile ctr_pipeline.pdf ctr_pipeline_preview
```

字体使用系统 Arial，并嵌入 PDF。SVG 使用可编辑文本。`extract_frames.py` 仅在重新取帧时需要原视频文件。
