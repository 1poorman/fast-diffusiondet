# Q1：最后一级质量感知分类与同预算H6对照

协议固定于2026-10-13，结果产生前记录。P1减框没有实用加速，G1诊断显示H4增加的
预测主要集中在中低分背景区域。本轮独立检验分类分数的定位质量监督，P300不变。

## 单一改动

方法参考[GFL的Quality Focal Loss](https://arxiv.org/abs/2006.04388)。本轮额外保留原focal的
正负alpha权重，并仅用于最后一级；没有实现GFL的分布回归，也不称为完整复现GFL。

最后一级将二值sigmoid focal替换为带原alpha权重的quality focal：

`L_cls = sum a(y) * BCEWithLogits(z, q) * abs(sigmoid(z) - q)^2 / max(1, N_positive)`

- Dynamic-K匹配算法、匹配代价、L1/GIoU、辅助头的原focal均不变。
- 匹配正框的GT类别目标q为预测框与匹配GT的detached IoU；其他类别和未匹配框q=0。
- `a(y)`按原二值正负归属取0.25/0.75，避免同时改变类别权重；不是把soft target直接
  代入旧focal的二值p_t公式。q=0/1时退化为原alpha/gamma=2的sigmoid focal。
- 对退化框按非负面积和epsilon计算IoU；目标不向框回归反传梯度。
- 不增加模型参数、质量分支、推理乘分或NMS变体。新配置默认关闭，旧权重严格加载。

## 三臂与预算

| 臂 | 结构与监督 | 训练预算 |
|---|---|---|
| h4_focal | 真实H4，原focal | 复用T1 uniform三份600步最终权重；本轮重新评测 |
| h4_quality | 真实H4，最后一级quality focal | 从原六级短训权重裁剪初始化，各600步 |
| h6_focal | 完整H6，原focal | 从同一原六级短训权重初始化，各600步 |

训练seed10/11/12；每步batch8、LR2.5e-5、AdamW、weight decay1e-4、clip1，
20步warmup(0.1→1)、480步LR×0.1，uniform t，KD/AMP/EMA关闭。
不从已经额外训练600步的H4开始新臂，避免引入额外训练预算。
偶数训练seed顺序h4_focal→h4_quality→h6_focal，奇数反转。

复用控制必须通过以下核验，否则停止并修复协议，不能直接比较：

1. T1完整记录、源码快照、旧初始化权重与三个控制checkpoint的SHA256。
2. 每seed在原初始化上重放第一批，原focal各项损失与T1记录一致。
3. 新H4初始state digest与T1一致；H6去掉最后两级后的初始digest也一致。
4. 新臂逐步图像/GT、填充框、eps、uniform t、私有gate、forward后全局RNG、LR，
   与T1对应训练seed记录一致。匹配结果和损失允许随新训练目标变化。
5. 重新评测的h4_focal逐seed、逐项AP与T1一致；本轮所有臂使用重新测量的延迟。

cuDNN训练deterministic开启，仍不宣称ROIAlign反向等全部算子逐bit可复现。
h6_focal与H4的差别包含两级检测头及辅助监督，比较的是同预算训练方案，
不能进一步分离容量与辅助监督的各自因果贡献。

## 评测与预设判定

每份最终模型评测750张SDD开发验证图、推理seed0/1/2，共27次全量评测。
GPU1、FP32、TF32关闭、NFE1、P300、800/1333、batch1，20图预热+同200图同步计时。
每个模型先平均推理seed，再跨训练seed比较；保留逐类AP、AP75、p50/p95和训练成本。

Q1相对同H4原focal：平均ΔAP≥+0.5、平均ΔAP75>0、至少2/3配对AP为正，
p50增幅≤5%、最差类别平均ΔAP≥−2.0。全部通过才列为开发候选，不按单个seed选型。
同预算H6对照另报H4−H6的AP/AP75和速度；不将这个比较算作quality focal的收益。
开发候选仍需独立数据确认；不在本轮搜索alpha、gamma或叠加时刻/KD/减框。

另用原G1诊断的固定分数阈值0.05/0.3/0.5，对保存的预测统计背景、错类、定位、重复框
及分数-IoU关系，检查原始假设。它不改变AP门槛，也不选择部署阈值；
覆盖率允许多个GT共享预测，不能视作COCO AR或TIDE的AP归因。

## 入口

`scripts/v3_quality.py`执行训练、控制重放和配对审计；
`scripts/v3_summarize_quality.py`核验并归档结果。
`scripts/v3_quality_diagnostics.py`执行冻结预测的描述性误差分析。
冒烟使用seed10、每新臂3步、16图评测，仅验证管线；LR仍沿用正式600步日程的前缀。
所有新权重和日志保存到新的`runs/v3/`目录，不覆盖历史权重。

## 完成记录（2026-10-14汇总）

正式实验`runs/v3/quality_20261013/`已完成：6个新模型各600步、3个复用控制、
27次750图评测，主进程exit code 0，GPU1已释放。配对输入/RNG/LR、初始化、
权重/源码/标注哈希与首批控制重放均通过；H4控制九次评测的全部AP与T1完全一致。
运行时冻结协议与源码保留在`source_snapshot/`；本节是结果产生后的记录，不修改上面的门槛。

**Q1未通过。** quality−focal平均ΔAP=−0.170、ΔAP75=−0.720，仅1/3配对AP为正；
p50+0.037%、最差类别−0.633 AP满足各自限制，但不足以通过完整门槛。
保持原focal。固定阈值诊断显示背景框下降也伴随正确匹配和覆盖率下降，不能视作AP提升。

独立同预算深度比较：H4/H6原focal为67.777/68.056 AP，p50为28.087/34.152ms；
H4−H6为−0.278 AP、+0.253 AP75，p50低17.76%。仅3对且速度不足20%，
不能宣称已达联合目标或获得独立确认。完整表格与解释见[RESULTS.md](RESULTS.md)。

复现汇总与冻结预测诊断，无需重新训练：

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_quality.py \
  runs/v3/quality_20261013
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_quality_diagnostics.py \
  runs/v3/quality_20261013
```
