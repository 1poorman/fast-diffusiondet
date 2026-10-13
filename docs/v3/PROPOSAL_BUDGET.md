# P1：固定候选框预算筛选

协议预先固定于2026-10-12，结果产生前记录。用户询问其他尝试后，选择与蒸馏、时刻
对齐不同的计算轴：减少每级ROIAlign、自注意力、DynamicConv处理的候选框数量。
本轮只筛选固定P；**不是已经实现逐级裁剪或完成训练补偿**。

## 固定比较

- 权重：T1中3份600步uniform真实H4学生，训练seed10/11/12全部纳入，不挑最好seed。
- P=300/200/150/100，每级同P，H4不变；冻结权重，不重新训练。
- 全部从同一(1,300,4)高斯噪声池取前P个框；池由既有`image_seed(eval_seed,image_id)`生成。
- pre-NMS top-k恒为300，NMS阈值不变；不将后处理预算缩减混入P的效果。
  每个框仍有7个独立分类分数，P=100也有700个分数供top-k选择。
- inference seeds0/1/2；750张全量SDD开发验证图；共36次评测。
- 先平均每个checkpoint的推理种子，再比较3份checkpoint的配对变化。
  它衡量已训练模型的P敏感性，不是重新训练了3份小P模型。
- FP32、TF32 off、batch1、NFE1、800/1333，20图预热+同200图同步计时。
  计时包含完整噪声池生成、取前缀和整次模型调用；不含CPU resize/evaluator/覆盖率诊断。
- 每份checkpoint先检查两张真实图：P300注入噪声的框/分数/类别与原路径逐bit一致。
  正式汇总还检查P300 AP与对应T1记录一致。推理种子与训练种子奇偶决定正/反遍历顺序。

## 召回与候选门槛

对每级NMS前的P个预测框记录：所有GT、逐类别GT、拥挤图GT（每图≥5个）的IoU0.5/0.75覆盖率。
框缩放回原图并裁剪到边界，按每个GT的最佳IoU统计，不要求预测类别；
允许一个预测覆盖多个GT，故这是**类别无关的候选覆盖率，不是COCO AR**。
原始AP/AP75/逐类AP仍按完整COCO协议评测，不能用覆盖率代替检测准确率。

小P相对同checkpoint P300的开发集筛选门槛：

1. 平均ΔAP ≥ −0.3，平均ΔAP75 ≥ −0.3。
2. p50平均下降至少10%。
3. 最差类别的平均AP下降不超过2.0；最后一级拥挤图IoU0.5覆盖率下降不超过1个百分点。

通过只列为速度候选，后续仍需独立数据确认。若失败，记录精度/召回代价，再考虑
先300框探索、后级压缩，并训练与推理保持一致；不事后放宽门槛或挑选有利seed。
G4完整1.4倍加速目标仍另行判断，本轮筛选通过不等于G4通过。

## 实现与复现

- `inference_noise`为显式评测输入，训练时禁止注入；None保持原采样路径。
- `INFERENCE_TOPK=0`保留旧逻辑，正值才独立控制输出预算；无新增模型参数。
- `diffusiondet/proposal_budget.py`负责共享噪声前缀；runner保存权重/源码/标注哈希与逐级覆盖率。
- 小规模GPU冒烟为seed10、eval seed0、32图、P300/100，仅用于验证管线。

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_proposal_sweep.py \
  --gpu 1 --out-dir runs/v3/proposal_sweep_20261012
```

相关替代方向：质量感知分类监督优先解决中低分背景框；分布回归单独验证AP75；
FP16/BF16或部署编译单独建立数值/延迟表，不能把精度模式改变归为架构收益。

## 正式运行与归档（2026-10-13接续）

2026-10-12的32图冒烟已完成；正式运行接续到新目录`runs/v3/proposal_sweep_20261013/`。
预先固定的模型、预算、种子、门槛均沿用上文。结果完成后执行：

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_proposals.py \
  runs/v3/proposal_sweep_20261013
```

汇总拒绝不完整矩阵、重复行、非有限AP、缺失逐类/逐级覆盖和P300不一致。
核验三份权重、源码快照、原始及评测用标注；先平均每个checkpoint的三个推理种子，
再对三份checkpoint等权平均。p50降幅先在checkpoint内计算比率，再取均值；
覆盖率变化用百分点，类别门槛使用该类别跨checkpoint的平均AP变化。
原始CSV、配对/逐类汇总、覆盖率、配置、manifest、逐bit对拍和Gate归档至
`results/raw/v3_proposal_sweep*`，图保存至`results/figures/v3_proposal_sweep.png`。

正式运行已完成36/36次评测，全部审计通过。P200/150/100平均ΔAP分别为
−0.303/−0.457/−0.999，p50降幅仅0.50%/0.71%/0.93%；三个候选均未通过。
保留P300，详细AP75、召回与逐类证据见[RESULTS.md](RESULTS.md)。
