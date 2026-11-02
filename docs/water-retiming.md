# 递水时序编辑

唯一入口为 `python -m real_robot_data_retime.water`。历史的 pilot/v4/v5/v6/v7、
重试脚本和修改源码生成脚本的流程已删除；历史实验及局部修补仍可从 Git 查看。
无需搬动或覆盖原数据集。

## 阶段和监督

- 根据冻结 action 的开夹爪命令（开口超过 20 mm、至少持续 0.2 秒）识别事件。
  每臂准备结束为第二次打开加 0.6 秒；同步结束为双臂释放托盘中较晚的打开加 0.2 秒。
- 两臂准备结束帧不同时，先各自结束独立动作，再监督剩余的源帧到共同同步起点。
  这段连接不计入独立动作时长。同步区间两臂使用相同源时钟。
- 复用仓库的静止检测裁掉每臂前置静止；默认阈值为 3 度/秒和 3 mm/秒，
  可通过 `--motion-threshold` 设置。记录每臂裁剪量及双臂共同静止前缀。
- 返回时长截止于冻结 action 的最后运动帧；之后保留原始收尾并补 1.5 秒终态。
- 只有后发臂的可选起步等待被 mask；起步边界、完成等待、同步及终态全部监督。
  整条 episode 至多一臂出现 idle，p=0 时两臂均无额外 idle。

所有运动按原控制序列逐帧推进，不引入旧版额外速度斜坡。时长按采样点间隔
计算，即 N 个点占 N−1 帧时长。冻结序列完整可读，因此没有平均时长估计。

## 手动采样

```bash
python -m real_robot_data_retime.water render \
  --source /path/to/raw --output /path/to/new-output \
  --episodes 20 --first-arm left --delay-ratio 0.5
```

p 在 [0,1]；返回与准备使用同一先发臂和 p。每阶段后发臂延迟为先发臂该阶段
实际时长乘 p，四舍五入到整帧。p=1 严格等到先发完成才开始后发。
输出为 output/dataset，工作视频、分割及审计位于 output/work。

## 均匀采样

```bash
python -m real_robot_data_retime.water plan \
  --source /path/to/raw --output /path/to/new-plan --samples 4

python -m real_robot_data_retime.water render \
  --source /path/to/raw --output /path/to/new-render --samples 4
```

第一独立阶段采用有符号偏移 Δ=右臂开始−左臂开始，在 [T左, −T右] 之间均匀取点，
包含两个严格串行端点。复用 retime.build_uniform_schedule_plan 的整数舍入规则。
全局格点 g=源序号+源数量×样本序号，总数=源数量×每源样本数，分母为总数−1。
不同源按各自冻结时长映射同一个归一化格点；可用 --episodes 选择源子集。

由准备阶段量化后的 Δ 确定先发臂与 p，返回阶段复用二者；返回阶段不再独立均匀
排列。收据记录请求偏移、实际整帧偏移、每臂时长、起止时间和 p。
当两臂时长不同，返回阶段不宣称严格均匀。

## 验证与缓存

每次 plan/render 必须指定全新输出目录，拒绝覆盖或仅凭路径复用旧视频。
新边界扩大了分割范围，必须重新生成 pre/post masks；分割文件包含其源帧覆盖区间。
检测歧义、前景冲突或导出错误直接抛出。输出保留 visual_review=pending，
数值与帧数校验通过不等于已完成人工视觉验收。发布不属于此入口。



## 实现来源与复核范围

此入口复用已有的水任务实现（源代码提交 `6733992`），替换发布仓库中只有调度核心、没有完整执行入口的旧状态。保留 `water plan|render` 单一入口，不保留旧批处理调度接口。

实际验证应以当前执行生成的 `VALIDATION.json`、源时间映射和视频为准，不把其他仓库的历史处理数量当作本次发布版本的运行结果。
