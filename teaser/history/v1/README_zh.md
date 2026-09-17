# CTR 论文首页图

打开 `ctr_teaser_editable.pptx` 即可使用组合后的完整主图，共一页。画布为 1760 × 870，约 2:1，适合横向论文图。

PPT 内有 13 个命名分组。每组可独立移动、缩放和复制；取消组合后，可以继续编辑文字、连线、曲线和图形。它们均为 PowerPoint 原生对象，整页没有合并成图片。选择窗格中的分组按阅读顺序命名。

本机 PowerPoint 当前提示尚未激活，因此只能预览。文件本身保留完整编辑能力，可在有编辑权限的 PowerPoint 中使用。

## 文件

- `ctr_teaser_editable.pptx`：可编辑的单页组合图。
- `ctr_teaser.svg`：完整主图，保留命名分组、可编辑文本和矢量路径。
- `ctr_teaser.pdf`：论文用矢量 PDF，默认宽度 175 mm。
- `ctr_teaser_preview.png`：由最终 PPT 渲染的高清预览。
- `modules/`：九个可独立使用的 SVG 内容模块。

## 独立 SVG 模块

| 文件 | 内容 |
| --- | --- |
| `original-causal.svg` | 抽屉任务的原始时序图 |
| `original-preference.svg` | Coca-Cola / Sprite 任务的原始时序偏好 |
| `retimed-causal.svg` | CTR 后保持必要顺序的时序图 |
| `retimed-preference.svg` | CTR 后展开的可行时序覆盖 |
| `ctr-process.svg` | CTR 处理模块 |
| `bc-before.svg` | 原始数据训练的 BC 与 Action |
| `bc-after.svg` | CTR 数据训练的 BC 与 Action |
| `binding-prerequisite-example.svg` | 抽屉条件变化时动作响应的例子 |
| `irrelevant-progress-example.svg` | 无关进度变化时局部动作保持的例子 |

SVG 模块均有透明画布，文字仍是文本。Inkscape 或 Illustrator 可编辑内部元素。插入 PowerPoint 后也可按软件版本提供的“转换为形状”功能处理；随附 PPT 已经是原生对象，无需这一步。

## 图意与术语

两类原始演示使用同样狭窄的时序曲线，表达仅凭演示时序无法区分必要先后关系与人为偏好。CTR 保留任务要求的时间关系，在独立任务中扩展合法的相对时序。右侧表达所希望得到的行为选择性：对绑定任务条件保持响应，对无关的另一只手进度近似不变。

曲线是联合进度空间中的示意轨迹，每条曲线代表一种相对时序。它们不是实测概率密度，也不是单任务边缘分布。“展开”表示更广的合法时序覆盖，不表示严格均匀概率密度或穷举连续空间中的所有轨迹。上方相同曲线示意必要关系保持，不声称所有有因果约束任务的完整概率分布都不变。

正式方法名称使用论文中的 **Counterfactual Temporal Recombination (CTR)**。

## 论文插图

```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/ctr_teaser.pdf}
  \caption{Demonstrations can confound task-required temporal relations with
  incidental scheduling preferences. Counterfactual Temporal Recombination
  preserves annotated task constraints while broadening admissible joint-progress
  coverage. The same behavior cloning learner can thereby reduce sensitivity to
  irrelevant partner progress while retaining the coordination required by the task.
  Trajectories and action responses are schematic.}
  \label{fig:ctr-teaser}
\end{figure}
```

## 来源

论文叙事来自 `/Users/shuyuan/Documents/CTR/Paper/main.tex`。轨迹几何、13 条对称单调曲线与 OKLab 渐变色场复用用户提供的 `generate_figure.py`；该源脚本保持不变。布局参考用户提供的 `retime.drawio.png`，图中示意模块由本次任务绘制。
