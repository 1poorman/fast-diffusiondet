# R1：ROI batch索引使用已知框数量

协议固定于2026-10-15，结果前记录。L1的H4/P300每图ROI池化包含28次stream同步：
16次nonzero、8次scalar读取、4次copy。本轮只替换pooler格式化时的batch索引构造，
目标是消除CPU repeat counts复制到GPU及repeat_interleave动态输出长度同步。
不改变FPN层分配、nonzero、ROIAlign、框坐标、层数或框数，不修改安装的Detectron2。

## 实现与正确性门槛

- 默认`ROI_POOLER_KNOWN_COUNTS=False`；可选本地ROIPooler子类，以Python shape数量
  创建各图的batch-index列。不读取GPU坐标，不删掉列表、金字塔层数或batch尺寸检查。
- 训练/推理都可选择该无参数pooler；checkpoint键保持兼容。JIT/FX tracing保留原路径。
- 先验证不等长batch、空图/空batch、单/多尺度、跨FPN阈值、越界框、格式张量的值/梯度。
  验证pooler特征输出与CPU梯度完全一致；CUDA特征梯度允许rtol=2e-5、atol=2e-5，
  因ROIAlign反向atomic累加可能非确定。完整head输出精确一致、参数梯度有限并一致。
- 冒烟仅seed10、推理seed0、32图；两臂完整预测digest必须一致，不使用子集AP作结论。

## 固定正式矩阵

T1的seed10/11/12三份600步uniform H4末步权重，P300、NFE1、原focal、no_weights，
全模型FP32、TF32/cuDNN benchmark/cuDNN deterministic关闭、4个CPU线程、batch1、GPU1。
原pooler/known-count pooler两臂，各推理seed0/1/2，共18次完整750图SDD开发验证。
共享300框噪声、固定top-k300，保持输入尺寸、NMS与所有其他计算。

记录全部预测有限、两臂每图最终预测digest完全一致、源权重与起止state digest一致。
九组原pooler的总体/逐类AP必须完整复现T1；候选的总体/逐类AP必须与配对控制一致。
源码、第三方poolers源码、权重、标注和预测/计时产物保留hash与快照。

## 延迟、机制检查与预设决策

完整验证内记录20图预热+随后200图的CUDA同步调用p50/p95与峰值allocated显存。
另外9组逐图AB/BA交错计时：按图序号+训练seed+推理seed奇偶切换次序，
相同图像/噪声、每臂20图预热+同200图计时。先平均推理seed，再等权平均checkpoint；
速度门槛使用checkpoint降幅的均值。包含噪声池生成及完整调用；不含读取/CPU resize、
设seed、evaluator和digest/数值检查。计时阶段不使用hooks或profiler。

所有计时完成后，再做三份权重×两臂×10图独立profiler，检查ROI域同步归属和repeat_interleave。
profiling预测与未插桩预测digest一致；profiler耗时不参与性能结论。
预期仅减少与batch索引构造有关的同步，不宣称消除了全部ROI同步。

只有同时满足才列为开发执行候选：

1. 全量逐图预测精确一致、全部AP控制复现、有限性/状态/源码审计及梯度检查通过。
2. 机制检查确认ROI repeat_interleave消失，ROI域同步次数减少。
3. 交错p50平均降低至少5%，每份checkpoint至少降低3%；平均p95增幅不超过5%。

不降低门槛，不叠加骨干FP16/QFL/减框。未达速度门槛则保持原pooler默认，
不把同步次数减少等同于可观的端到端提速。即使通过仍需未用于选型数据与目标硬件确认。
所有新运行产物使用新的`runs/v3/`目录，旧权重只读。
