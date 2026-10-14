# L1：冻结模型的分段耗时审计

协议固定于2026-10-14，运行前记录。Q1已补齐同预算H6；P1减框损失AP但提速不足1%。
本轮确认当前FP32实现的成本分布，为下一项独立优化提供证据，不训练模型、不重算子集AP。

## 固定矩阵

- 三份T1 uniform H4及三份Q1 H6 focal，训练seed10/11/12全部保留。
- H4/P300、H4/P100、H6/P300；推理seed0/1/2，27个单元。这里只比较P端点解释P1现象。
- GPU1、RTX3090、FP32、TF32 off、NFE1、800/1333、batch1、4个CPU线程，cuDNN benchmark/deterministic关闭。
- 沿用排序后的SDD验证集前20图预热及随后200图计时；不是全验证AP。
- 完整300框噪声池取前P个，pre-NMS top-k恒为300，NMS和权重不变。
- 奇偶训练seed+推理seed反转三臂顺序；每个单元先无插桩、再分段计时。
  全部27个单元计时完成后才启动算子profiler，避免CUPTI状态影响后续无插桩基准。

## 三种记录及解释边界

1. **无插桩基准**：完整噪声池生成+模型调用，前后CUDA同步，记录逐图wall time、p50/p95/均值。
   读取、CPU resize、生成随机种子和输出校验在计时外，口径沿用P1。
2. **分段记录**：临时包装预处理、backbone、head、每级头、ROI池化、attention、DynamicConv、
   框解码、top-k/NMS和输出后处理。记录嵌套的CPU调用区间和同一CUDA stream上的event区间；
   只在整次模型调用末尾同步，不逐段同步。按直接子区间相减得到exclusive区间，再按图求和/均值。
   CUDA event区间包含GPU执行和CPU供给间隙，CPU调用区间也可能含同步等待；二者都不是纯计算时长。
   报告插桩相对无插桩的开销，只用无插桩数据报告实际响应时间；不相加各段p50，不加父子区间。
3. **算子轨迹**：所有三份checkpoint的三臂，固定推理seed0、计时区间前10张图，
   单独使用PyTorch CPU/CUDA profiler；保存Chrome trace、算子self时间、CUDA kernel/复制区间与同步调用。
   轨迹有显著扰动，只用于辨识实际算子和同步机制，不以profiler时长当部署延迟或推算确定加速。

每个计时图均核验分段包装前后的框/分数/类别digest一致；profiler输出也与对应无插桩输出核对。
检查head与每级调用次数、实际P、完整图像矩阵、权重/源码/配置/标注哈希；错误则不归档结论。
使用临时包装并在异常时恢复，不修改检测器生产前向或Detectron2安装文件。
大轨迹与逐图区间保留于`runs/v3/`；小型汇总和图归档到`results/`。

## 后续决策

本轮是描述性成本诊断，没有模型晋级门槛，也不更改P1/Q1的失败判定。
若框数减少只缩短kernel而总延迟几乎不变，结合调度/同步轨迹决定是否先做执行优化；
若主要成本在backbone，则优先独立考虑骨干或精度模式。任何后续实现需另做输出兼容性、
无插桩速度及完整AP验证，不能把成本占比直接当作可实现的提速比例。

## 完成记录

正式目录`runs/v3/latency_20261014/`已正常完成，27个单元、54份计时记录、9组算子轨迹。
5400次event对拍和90次profiler对拍通过；完整证据审计通过，GPU1已释放。
13项新增审计测试及已有检测头/候选预算回归共26项通过。
运行时协议与源码快照保留于`source_snapshot/`；本节为结果产生后的记录。

本轮无插桩H4/P300、H4/P100、H6/P300的p50均值为28.645/27.473/33.238ms。
P100相对P300平均下降4.10%，单轮范围−10.17%至+12.50%；不能替换P1的原始结果，
也不能据此声称稳定实用加速。event与plain固定先后次序，其差值同时含插桩与时间漂移，
本轮甚至出现负差值；不把差值当作纯插桩开销或通过插桩得到的加速。

H4/P300的backbone约占总stream区间41%，完整head约48%；ROI同步每图28次，
P100仍28次，H6为42次。下一步分别验证骨干精度模式和ROI动态长度同步的优化，
不由区间比例直接预测加速。详细表格、误差边界和产物见[RESULTS.md](RESULTS.md)。

```bash
# 重新测量须使用新的输出目录，避免覆盖本轮证据。
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_latency_profile.py \
  --gpu 1 --out-dir runs/v3/latency_new

# 从本轮证据复现汇总，无需GPU或重新推理。
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_latency.py \
  runs/v3/latency_20261014
```
