# Beyond Demonstration Schedules — ICLR 2027 LaTeX 作者工作包

## 使用

将整个文件夹压缩包上传到 Overleaf，选择 `main.tex` 为主文档，编译器选 **pdfLaTeX**。所有表格与 PDF 图件均已生成，正常编译论文不需要安装 Python。默认匿名审稿格式；未启用 `iclrfinalcopy`。

本地完整构建：

```bash
bash scripts/compile.sh
```

安装了标准 `latexmk`/BibTeX 的环境也可用：

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

重新生成数据表和图件：

```bash
python3 -m pip install -r requirements.txt
python3 scripts/build_results.py
python3 scripts/make_figures.py
bash scripts/compile.sh
```

## 内容

`main.tex`：英文正文、声明与文献入口。方法暂名 **Counterfactual Temporal Recombination (CTR)**，可由作者更名。

`appendix.tex`：任务、指标、实现字段、核验说明、完整逐条件结果。

`references.bib`：已核对的 13 项相关文献；`plainnat` 生成 author–year 引用。`iclr2027_conference.sty`：官方样式文件，未修改。

`figures/`：两张由实际汇总结果绘制的干预图，以及 TikZ 方法示意图。示意图不是实验图像。

`tables/`：自动生成的 5 张正文表与 6 张详细结果表。

`data/experiments.json`：上传的原始 JSON 的逐字节副本；5 个 CSV 是其表格化导出。`derived_summaries.json` 为正文所用描述性汇总。原始数据不被自动修正。

`scripts/`：图表重建、统计口径审计与编译脚本，不包含策略训练或机器人控制实现。

`notes/AUTHOR_CHECKLIST_zh.md`：提交前必读。`notes/data_audit.json`：条件性整数计数检查和区间宽度清单。`notes/claim_provenance.json`：核心数字的筛选与聚合口径。`data/implementation_metadata.template.json`：等待作者填写的实际配置。

`main.pdf`：本次已编译预览。

## 当前状态

稿件包含完整叙事与实测结果，不是空白模板。但只有汇总表，没有逐种子/逐回合日志和全部实现配置，因此不是已完成事实核验的投稿终稿。原始区间仅按输入转录，未计算 p 值、显著性或统计等效；不把不同任务/条件的结果拼为虚构回合。

稿件明确区分已提供结果、按约定指标写出的协议、待核验字段与文献背景。请用实际实验配置关闭待核验项，不要仅删除提示。修改 JSON 后，表格与图可以重建；正文叙述数字也必须同步核对。

## 模板来源

官方 ICLR 模板仓库的 `iclr2027/iclr2027_conference.sty`：
https://github.com/ICLR/Master-Template/blob/master/iclr2027/iclr2027_conference.sty

取得时 Git blob SHA：`f61ad7efce0855557694078c0945e6c33feb8236`，包内副本与之相符。官方初次提交正文限 9 页；参考文献与附录不计入。当前正文为 8 页、总计 24 页。实际要求请以投稿时官方作者指南为准：
https://iclr.cc/Conferences/2027/AuthorGuidelines

已加入 required AI-use statement，真实工作流披露需由作者最终确认：
https://iclr.cc/Conferences/2027/AIPolicyForAuthors

包内没有字体文件。PDF 的正常内嵌字体不属于单独提供的字体资源。
