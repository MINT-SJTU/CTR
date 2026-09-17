# 本次叙事重构说明

本版本按“第二篇的叙事骨架 + 第一篇 CTR 的实际实现与最强 diagnostics”重写。

主要变化：

1. 标题改为 **Learn the Task, Not the Schedule: Removing Demonstrator-Induced Scheduling Bias in Bimanual Imitation Learning**。
2. 将论文中心从“一个 bimanual data augmentation 方法”前移为“imitation learning 的 failure mode”：demonstration 同时编码 task 与 demonstrator schedule，policy 会把后者误学为 task constraint。
3. 保留并前置 **What is and is not identifiable**，明确：仅从 demonstration 无法区分 prerequisite 与任意 serial scheduler；CTR 使用训练时约束标注，不宣称从轨迹恢复隐藏 causal graph。
4. CTR 的实现保持为第一篇：保留 arm-local motion，只改变 admissible relative timing；部署时不需要 scheduler。
5. 加入实际实现：**GPT-6 Astra** 标注可重排/必须保护的 temporal segments；**RobotSeg** 等开源工具识别机械臂与任务物体并做前景分割，合成重排后的主视角视频。
6. 按指定方式组织 real-data implementation：
   - Issue 1: reconstruct temporally consistent observations → visual composition；
   - Issue 2: avoid treating synthetic waits as expert decisions → IdleMask。
7. 实验叙事改为 failure-mode 证据链：
   - narrow schedule support 的 completion-state brittleness；
   - paired partner-arm pause 作为最直接 behavioral diagnostic；
   - held-out joint-progress generalization；
   - prerequisite/workspace release 检验 selective coordination；
   - ablation 回答 temporal variation / IdleMask / main view 各自作用。
8. 保留原始数据文件、主表和详细结果数值；`scripts/build_results.py` 仍可从原 aggregate export 重建表格。
9. 删除主文中大量“不是 X / 不能说明 Y”式防御性 caveat；必要的边界集中到 identifiability 与一段 Limitations 中。
10. 已重新编译 `main.pdf`，无 undefined references/citations，也无 overfull box。
