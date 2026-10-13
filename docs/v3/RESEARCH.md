# V3 技术检索与迁移判断

检索日期：2026-10-11。阅读了 arXiv 摘要、官方项目说明；不是全文复现。
检索词覆盖 real-time object detection、RF-DETR、DEIM/DEIMv2、D-FINE、
RT-DETRv3 和 DiffusionDet distillation。以下以论文原页为依据，不以转载排行选型。
跨论文的 COCO AP、T4/TensorRT FPS 不能直接作为本项目 SDD/3090/PyTorch 的预期值。

| 来源 | 已核实的方法 | 对本仓库的具体启发 | 采用次序 |
|---|---|---|---|
| [RF-DETR: Neural Architecture Search for Real-Time Detection Transformers](https://arxiv.org/abs/2511.09554)，2025，2026 修订；[官方代码](https://github.com/roboflow/rf-detr) | 共享权重的架构搜索，在目标数据集上找精度—延迟 Pareto 曲线 | 把实际检测头层数、proposal 数、输入尺寸当作资源旋钮。先评测已有辅助头输出；这不等于已经实现 RF-DETR 的超网训练 | P0 |
| [D-FINE](https://arxiv.org/abs/2410.13842)，2024；[官方代码](https://github.com/Peterande/D-FINE) | Fine-grained Distribution Refinement（FDR），Global Optimal Localization Self-Distillation（GO-LSD） | 用更深层定位知识训练浅层；进一步把 4 坐标回归改为离散边界分布，服务 AP75 与浅层提前输出 | P1；分布回归 P2 |
| [DEIM](https://arxiv.org/abs/2412.04234)，2024；[官方代码](https://github.com/ShihuaHuang95/DEIM) | Dense O2O 增加正匹配，Matchability-Aware Loss 处理低质量匹配 | 本仓库已经用 Dynamic-K 多正样本，不能把“更多正样本”当成缺失功能直接照搬。优先考虑定位质量与分类分数对齐，以及受控的数据增强 | P1/P2 |
| [RT-DETRv3](https://arxiv.org/abs/2409.08475)，2024；[官方代码](https://github.com/clxia12/RT-DETRv3) | 训练专用 CNN 辅助分支、自注意力扰动、共享解码器分支提供密集正监督 | 若语义特征是瓶颈，给 FPN 加训练专用 dense branch；不能宣称额外训练零成本，但部署可移除该分支 | P2 |
| [DEIMv2: Real-Time Object Detection Meets DINOv3](https://arxiv.org/abs/2509.20787)，2025；[官方代码](https://github.com/Intellindust-AI-Lab/DEIMv2) | DINOv3 预训练/蒸馏骨干、Spatial Tuning Adapter 恢复多尺度；小模型使用剪裁 HGNetv2；简化 decoder | 增强语义应兼顾多尺度细节。当前 R50+FPN 已有多尺度，先验证局部头瓶颈，再考虑骨干替换或教师特征蒸馏 | P2/P3 |
| [EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation](https://arxiv.org/abs/2603.18739)，2026；[官方项目](https://intellindust-ai-lab.github.io/projects/EdgeCrafter/) | 面向检测任务的教师蒸馏到紧凑 ViT；轻量卷积 stem、插值/线性投影构建多尺度 | 教师应先适应检测任务，而不是直接匹配通用语义特征。可作为第二阶段骨干蒸馏或独立强基线 | P3 |

2026 年技术并非只强调更少扩散步，重点已包括检测专用蒸馏、紧凑骨干和可调整解码器。
对本项目，迁移成本最低的交集是**沿检测头深度压缩计算，再用训练监督补偿定位质量**。
保留原 R50/FPN 先做隔离实验，避免把预训练数据、骨干容量、输入分辨率和损失变化混在一起。

## 不直接照搬的部分

1. **D-FINE 的 GO-LSD 不是普通坐标 L1 蒸馏。** 当前模型没有边界分布；先做的坐标/分类蒸馏只能称为其启发的简化原型。分布输出、支持区间、残差迭代和分布监督落地后才能讨论 FDR 类方案。
2. **DEIM 的 O2O 问题与当前 Dynamic-K 不同。** 当前 `loss.py` 每图可把多个 proposal 分给同一个 GT，应首先统计正样本数和匹配质量，再决定是否需要密集监督。
3. **基础模型骨干不会自动提速。** DINOv3 大教师只用于训练；若直接部署大 ViT，应作为精度上界而非速度方案。
4. **公开 FPS 不作为收益承诺。** EdgeCrafter 项目明确使用 T4、batch=1、FP16、TensorRT 10.6，本轮为 3090、FP32、PyTorch，比较仅限同一实验协议内。
5. **生成模型的一步蒸馏未必适合当前主要误差。** 本仓库多步教师只比单步略好，优先蒸馏 6 级检测头到 3/4 级，而不是继续蒸馏扩散 NFE=2 到 1。

检索与设计之间的推断、目标阈值、尚未验证的迁移均在 [BLUEPRINT.md](BLUEPRINT.md) 单独列出。
