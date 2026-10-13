# V3 验证记录

日期：2026-10-11。主实验全部完成，由 `scripts/v3_summarize.py` 归档。

## G0：实现正确性

- `tests/test_v3_head.py`：4 passed。
- `tests/test_m1_ddim.py tests/test_m1_convergence.py`：11 passed。
- 覆盖提前退出实际跳层、前缀精确等价、训练保留全部层、注意力优化参数/梯度/数值兼容、旧采样器回归。
- GPU 冒烟：现存 SDD 权重严格加载，32 图、H6 legacy / H4 no_weights 都完成推理和 COCO 评测。
  子集不覆盖全部类别，**不将它的 AP 用于选型或宣传增益**。

## G1：冻结权重检测头扫描

主实验：`runs/v3/head_sweep_20261011/`，15/15 组完成，评测循环总耗时约 484 秒。
全部 AP 来自完整 750 图。3 个种子为推理噪声种子，不是训练重复。
使用短训 SDD bs8 ckpt，不能与历史 long ckpt 的 70.27 AP 比较。

### 协议与原始证据

- GPU1 / RTX 3090；torch 2.1.2+cu121，detectron2 0.6。
- 同一冻结 ckpt，NFE=1，P=300，FP32，TF32 关闭，输入短边 800/长边上限 1333。
- batch=1；每臂 20 图 warmup + 同一 200 图计时；CUDA 同步的模型调用 wall time。
  不含磁盘解码/CPU resize/evaluator，含模型内部预处理、H2D、backbone、检测头和后处理。
- 表中 AP 标准差为 3 个推理 seed 的 sample std（ddof=1）；延迟为各 seed 的 p50/p95 均值。
  Speedup 为按 seed 配对的 baseline p50 / arm p50 再求均值，不是批量吞吐实测。
- ckpt SHA256：`a79a15b0c3c2307fecd03d0d059c6fb24bb3c15786f3b9ceb6ffff59a4a9fc6f`。
- [逐次 CSV](../../results/raw/v3_head_sweep.csv)、[汇总 CSV](../../results/raw/v3_head_sweep_summary.csv)、
  [逐类 AP](../../results/raw/v3_head_sweep_per_class.csv)、[运行 manifest](../../results/raw/v3_head_sweep_manifest.json)、
  [数据审计](../../results/raw/v3_sdd_data_audit.json)。原始预测/配置在 run 目录。

### 结果

| H | Attention | AP (mean±sample std) | ΔAP paired | AP75 | p50 ms | p95 ms | Speedup |
|---|---|---|---|---|---|---|---|
| 6 | legacy | 66.540±0.117 | +0.000 | 72.301 | 34.64 | 37.60 | 1.000× |
| 6 | no_weights | 66.540±0.117 | +0.000 | 72.301 | 33.72 | 36.57 | 1.027× |
| 5 | no_weights | 66.200±0.055 | -0.340 | 71.912 | 30.77 | 33.38 | 1.126× |
| 4 | no_weights | 65.853±0.246 | -0.687 | 72.182 | 27.88 | 29.90 | 1.243× |
| 3 | no_weights | 64.622±0.184 | -1.918 | 71.394 | 24.95 | 27.09 | 1.388× |

![Head depth Pareto comparison](../../results/figures/v3_head_sweep.png)

### Gate 与设计更新

1. **G1 未通过，不能把 4 级头直接作为无损替代。** 4 级头平均 p50 降低约 19.5%，
   但 AP 配对下降 0.687（各 seed 为 −0.851 至 −0.566），超过 −0.3 门槛；
   速度也略低于预先设定的 20% 降延迟门槛。3 级更快，但精度缺口更大。
   5 级的 AP 均值下降 0.340，也未达到 −0.3 门槛。
2. **关闭 attention weights 的数值兼容性得到支持，独立加速证据较弱。**
   同 6 级逐 seed ΔAP 的绝对值均 <0.00001 AP；平均延迟略低，但配对增益随运行顺序有变化。
   不把约 2.7% 的平均加速认定为跨机器稳定收益。该优化继续保留为可选项，默认仍为 legacy。
3. **继续优先验证 H4 学生的精度补偿。** 与 H3 相比，H4 所需恢复 AP 更少。
   H4 至少补约 0.39 AP 才进入基线 −0.3 带，补约 0.69 才打平，补约 1.19 才达到基线 +0.5。
   当前没有训练证据证明这些目标能达到。
4. **不要盲目把所有框蒸馏自最后一层。** H4 的 AP50 比基线低约 1.02，AP75 仅低约 0.12；
   这提示应检查召回、分类/排序和不同 IoU 阈值误差，而非直接假定纯坐标回归能力不足。
   逐类差异也大：Powdery Mildew Fruit −3.120 AP，Blossom Blight −1.131，
   Gray Mold 反而 +1.230。下一步先做 GT 匹配的质量筛选蒸馏，重点检查退化类别。
5. 本原型保留全部 6 级参数以兼容权重，仅少执行后续计算；峰值 allocated 显存约 0.678 GB，
   **未观察到显存收益**。真正保存精简学生时再删除部署不使用的层。

### 下一步最小训练对照

固定相同 H4 学生初始化与预算，分别训练无 KD 和有质量筛选 KD；以同一 6 级教师作为参照。
先 200–500 iter 检查可训练性与各类方向，再至少 3 对完整训练 seed 判定。
分布回归、质量校准、proposal schedule 均按 BLUEPRINT 单独消融，当前尚未实现或运行。
**本轮结论是“压缩收益和精度缺口已量化”，不是“已经同时提升 AP 和速度”。**

复现汇总与图：

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize.py \
  runs/v3/head_sweep_20261011
```

## G2 初筛：四级学生质量筛选蒸馏（2026-10-11）

用户继续后，已完成独立冻结六级教师 → 真实四级学生的实现、26 项自动测试、
实际 SDD 权重裁剪输出对拍，以及 **3 对训练 seed × 每臂 200 步 × 每模型 3 次全量评测**。
共 6 个训练模型、18 次 750 图评测。详细损失与配对协议见 [DISTILLATION.md](DISTILLATION.md)。
这是从同一 SDD checkpoint 继续微调的短预算筛选，每臂仅 1600 图次，不能解释为从头训练结果。

### 配对结果

每个训练 seed 的 AP 先对 3 个推理 seed 求平均，再求下面的配对差值。

| Training seed | Control AP | KD AP | Paired ΔAP | ΔAP50 | ΔAP75 |
|---|---|---|---|---|---|
| 10 | 66.444 | 66.195 | -0.249 | -0.351 | -0.899 |
| 11 | 66.707 | 66.108 | -0.599 | -0.069 | -0.787 |
| 12 | 65.287 | 66.484 | +1.198 | +0.502 | +1.394 |
| Mean | 66.146 | 66.263 | +0.117 | +0.027 | -0.097 |

配对 ΔAP 的 sample std = **0.952 AP**。Control/KD 的训练 seed 均值标准差分别为
0.756/0.197 AP；只有 3 对，不能据此认定 KD 稳定地降低了训练方差。

![Paired KD pilot](../../results/figures/v3_kd_pilot.png)

### 部署与训练成本

| 项 | Control | KD |
|---|---|---|
| 部署结构 | 真四级 | 真四级；教师不部署 |
| 参数量 | 83,077,868 | 83,077,868 |
| 评测 p50 均值 | 27.81 ms | 27.91 ms |
| 评测 p95 均值 | 29.99 ms | 29.90 ms |
| 推理 peak allocated | 0.578 GB | 0.577 GB |
| 每臂训练耗时均值 | 138.54 s | 183.05 s |
| 训练 peak allocated 均值 | 7.26 GB | 12.08 GB |

原六级为 110,560,770 参数、G1 p50=34.64 ms。四级参数少约 **24.9%**，本轮部署延迟约
减少 **19.4–19.7%**，推理 allocated 显存约少 **14.8%**。KD 比 Control 训练耗时多约 **32.1%**，
这些训练成本包含被冻结教师的前向；不是免费的增强。

### 正确性与诊断证据

- 三对初始学生 state digest、逐步训练输入/增强 digest、图次数均相同。
- 三个教师起止 state digest 不变，梯度全部为 None；没有迁移初始化静默缺失。
- 每批平均约 45.88 个 KD 配对，约占学生正匹配的 66.5%；合格教师/学生平均 IoU=0.9394/0.9170。
  KD 不是因为完全无正样本而失效；也不能把这样小的平均定位差当作必然有效的监督。
- KD 最后 20 步平均加权分类/框损失为 0.0502/0.1516；额外损失实际参与反向。
- 逐类 KD−Control：Blossom Blight +1.540，Powdery Mildew Leaf +0.578，
  Powdery Mildew Fruit +0.338；Angular Leafspot −0.668、Anthracnose Fruit Rot −0.670，
  Gray Mold −0.217、Leaf Spot −0.085。收益不是普遍提升。

### 判定和后续优先级

**G2 初筛：证据不足，不晋级为默认模型。** 平均增益 +0.117 低于预设 +0.5 AP，
且 3 对中有 2 对下降。KD 相对旧六级均值低 0.278，名义进入 −0.3 AP 带，
但这不能替代 KD 相对同预算 Control 的收益判定，也没有达到联合速度门槛。
本轮没有证明“蒸馏稳定增强精度”。

保留真实四级结构、严格加载、配对训练和可选 KD 的实现。下一阶段优先：

1. **先稳定学生微调**：比较更长的固定微调预算与当前 200 步预算，仍用配对训练 seed，
   避免把近一个 epoch 的数据顺序波动误判为架构效果。需要对比六级同预算控制。
2. **再分离训练时刻对齐因子**：在四级学生上单独比较 uniform 与 mix_tmax，检查
   训练监督是否与单步 t=999 的部署目标对齐。保持 KD 关闭作为明确控制；旧 V1 只提供方向线索。
3. 若继续 KD，先分解分类/框 KD，并记录教师置信度与按 t 分桶的覆盖率。
   当前联合损失不直接扩大预算，也不因为 seed12 的好结果选择性宣布通过。
4. 分布回归和 proposal schedule 仍为后续独立改造，尚未实现。

### 可复现实验产物

- [18 次评测 CSV](../../results/raw/v3_kd_pilot.csv)
- [三对训练差值](../../results/raw/v3_kd_pilot_paired.csv)
- [训练成本与校验摘要](../../results/raw/v3_kd_pilot_training.csv)
- [逐类 AP](../../results/raw/v3_kd_pilot_per_class.csv)
- [manifest](../../results/raw/v3_kd_pilot_manifest.json) 与 [实际配置](../../results/raw/v3_kd_pilot_config.yaml)
- [物理裁剪对拍](../../results/raw/v3_student_export_check.json)
- 原始模型/输入轨迹/预测/源码快照：`runs/v3/kd_pilot_20261011/`。
  所有最终 checkpoint 均保留，不按开发集结果挑选单一“最佳 seed”作为整体结论。

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_distill.py \
  runs/v3/kd_pilot_20261011
```

## G1 冻结预测的误差补充诊断

在 T1 训练期间离线分析原 G1 的 H4/H6 no_weights、各3个推理 seed；未重训，
未使用真实标注改变模型分数。固定分析分数阈值0.05/0.3/0.5，不挑选部署阈值。
同类别 IoU≥0.5 按分数贪心一对一匹配；剩余框按重复、错分类、定位、背景依次归类。
这里的覆盖率允许多个GT共享一个预测，**不是COCO AR**；错误数量也不是TIDE的AP归因。

score≥0.3，每750图的3推理seed均值：

| 后处理结果 | H4 | H6 |
|---|---:|---:|
| 保留检测框 | 1800.33 | 1767.33 |
| 正确一对一匹配 | 1488.67 | 1483.33 |
| 重复框 | 5.00 | 3.67 |
| 错分类（与任意GT IoU≥0.5，正确类别未达0.5） | 20.33 | 19.67 |
| 定位偏差（同类别 IoU∈[0.1,0.5)） | 101.00 | 101.00 |
| 背景/其他低重叠框 | 185.33 | 159.67 |
| 同类别GT覆盖率，IoU≥0.75 | 76.53% | 75.81% |

score≥0.05 时，H4/H6 分别保留8878.67/6881.67个框，IoU0.5覆盖率97.58%/97.26%。
score≥0.5 时，两者背景框41.33/40.33，覆盖率70.33%/71.23%。因此 H4 额外输出
主要集中在较低置信度区域；不能只依据 AP 下降就认定它缺少框定位能力。
按置信度分桶的平均同类别 IoU 曲线很接近；score≥0.3且同类别IoU≥0.1的框中，
score-IoU Spearman为0.522/0.515，未显示显著的整体质量排序恶化。

**对蓝图的约束：**先考虑区分中低分背景的质量/分类监督；更复杂的分布回归仍需
证明能改善高IoU定位。NMS后的重复框较少，不优先改NMS。以上是开发集描述性现象，
不能估计某个新分支会增加多少AP，也不能证明提前裁剪proposal安全；后者必须测早期头召回。

- [诊断CSV](../../results/raw/v3_error_audit_diagnostics.csv)、[分桶数据](../../results/raw/v3_error_audit_calibration.csv)
- [数据与源码哈希](../../results/raw/v3_error_audit_manifest.json)、[诊断图](../../results/figures/v3_error_audit.png)
- 复现：`python scripts/v3_error_audit.py --out-dir runs/v3/error_audit_new`。

## T1：600步训练时刻对齐（已完成，不晋级默认）

实验于2026-10-11完成，2026-10-12汇总。6个真实H4模型、18次全量评测均正常退出，
`COMPLETE.json` 存在。两臂均无KD，每臂600步×batch8；协议与门槛预先固定于
[TIME_ALIGNMENT.md](TIME_ALIGNMENT.md)。每个表格点先平均3个推理seed，再比较训练seed。

| 训练seed | Uniform AP | Mix tmax AP | 配对 ΔAP | 配对 ΔAP75 |
|---|---:|---:|---:|---:|
| 10 | 67.534 | 68.107 | +0.574 | +0.525 |
| 11 | 68.042 | 68.033 | −0.009 | −0.703 |
| 12 | 67.756 | 68.588 | +0.832 | −0.304 |
| **均值** | **67.777** | **68.243** | **+0.465** | **−0.160** |

配对ΔAP的训练seed样本SD=0.431；模型内推理seed已先平均，不能把18次评测视作18次训练。
两组训练seed均值AP的样本SD分别0.255/0.302；只有3对，而且共享同一预训练初始化，
不据此宣称普遍的稳定性提升。

| 指标 | Uniform | Mix tmax |
|---|---:|---:|
| AP50 | 88.909 | 88.848 |
| AP75 | 74.103 | 73.943 |
| p50 模型调用延迟，ms | 27.625 | 27.656 |
| p95 模型调用延迟，ms | 29.479 | 28.466 |
| 推理峰值 allocated，GiB | 0.578 | 0.577 |
| 每模型600步训练时间，s | 407.95 | 405.06 |
| 训练峰值 allocated，GiB | 7.260 | 7.259 |
| 实际 t=999 占比 | 0.160% | 50.153% |
| 实际 t 均值 | 501.00 | 750.76 |

**判定：T1 未通过完整筛选门槛。** ΔAP≥+0.3、2/3种子为正、p50增幅≤3%均满足；
但平均ΔAP75=−0.160，低于预设−0.1底线，且2/3种子的AP75下降。
不能因为总体AP增加而事后放宽定位门槛。保持uniform为默认训练分布，mix_tmax保留可选实验。
这项训练改动没有减少部署计算；观察到的p50差异仅+0.11%，不能当作新增提速。

逐类收益并不均匀：Blossom Blight +1.931、Powdery Mildew Leaf +0.734、Angular Leafspot +0.725；
Anthracnose Fruit Rot −0.210、Powdery Mildew Fruit −0.109、Leaf Spot −0.085。
不能把提升解释为所有类别、所有定位阈值都获益；small目标仅12个，APs不作为主要依据。

### 已获得的实际速度—精度候选

| 配置 | AP | p50，ms | 额外微调 |
|---|---:|---:|---|
| G1 原六级 legacy 冻结参考 | 66.540 | 34.640 | 无 |
| G1 冻结权重四级出口 | 65.853 | 27.882 | 无 |
| T1 真实四级、uniform | 67.777 | 27.625 | 600步 |
| T1 真实四级、mix_tmax | 68.243 | 27.656 | 600步 |

600步uniform H4 相对原六级冻结参考，观察到AP高1.237、p50低约20.25%，参数少24.9%。
这是**裁剪、执行优化及额外微调的组合结果**，未设置同预算六级训练臂，不能把全部AP增益
归因于压缩，也不能据此宣布完整架构门槛已获独立确认。

相比G2的200步无KD控制均值66.146，本轮600步uniform均值67.777，高约1.631 AP。
三组seed前200步的增强图像/GT哈希各200/200一致，但LR衰减点从160移到480，
因此这是预算与学习率日程的共同变化，不能称为只增加步数的严格单因子实验。
它支持优先保留足够微调预算，再验证复杂损失的附加价值。

### 审计与后续蓝图

- 3对初始化、增强输入、x_start/填充框、eps、base t、私有gate、forward后全局RNG均一致。
- 逐图实际t符合配置；6份checkpoint、12份源码快照哈希核验通过；配对配置仅t分布/输出目录不同，
  逐步LR完全相同，日志无KD损失。47项测试通过。任务exit code 0，结束后GPU1显存0 MiB。
- 保留全部6份最终模型；未依据开发集挑选单一“最佳seed”，未将小数点附近的AP75失败改判为通过。
- 下一优先级为 **同预算H6对照**，确认压缩的代价；随后独立验证背景/定位质量监督。
  暂不搜索更多mix概率。分布回归与逐级proposal裁剪仍未实现，需各自的隔离实验与召回审计。

产物：
[18次评测](../../results/raw/v3_time_alignment.csv) ·
[配对汇总](../../results/raw/v3_time_alignment_paired.csv) ·
[训练审计](../../results/raw/v3_time_alignment_training.csv) ·
[逐类AP](../../results/raw/v3_time_alignment_per_class.csv) ·
[门槛判定](../../results/raw/v3_time_alignment_gate.json) ·
[manifest](../../results/raw/v3_time_alignment_manifest.json) ·
[配置](../../results/raw/v3_time_alignment_config.yaml) ·
[图](../../results/figures/v3_time_alignment.png)。
原始权重、预测、逐步日志及源码快照：`runs/v3/time_alignment_20261011/`。

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_time_alignment.py \
  runs/v3/time_alignment_20261011
```

## P1：固定候选框预算（2026-10-13完成，保留P300）

接续上次32图冒烟，本轮完成三份T1 uniform H4权重 × 四种P × 三个推理seed，
共36次750图全量评测。权重全部冻结，固定共享300框高斯噪声池的前缀，
pre-NMS top-k恒为300；协议和门槛预先固定于[PROPOSAL_BUDGET.md](PROPOSAL_BUDGET.md)。
这衡量已训练H4模型的预算敏感性，未训练小P模型，也未实现逐级剪框。

| 每级P | AP | 配对ΔAP | AP75 | 配对ΔAP75 | p50 ms | p95 ms | p50降幅 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 300 | 67.777 | +0.000 | 74.103 | +0.000 | 27.774 | 28.693 | 0.00% |
| 200 | 67.474 | −0.303 | 73.566 | −0.538 | 27.634 | 28.646 | 0.50% |
| 150 | 67.320 | −0.457 | 73.321 | −0.782 | 27.577 | 28.511 | 0.71% |
| 100 | 66.778 | −0.999 | 72.510 | −1.593 | 27.515 | 28.587 | 0.93% |

每个checkpoint先平均三个推理seed，再跨三份checkpoint等权平均。
p50降幅先计算checkpoint内的均值比率，再取均值。计时包含完整噪声池生成与模型调用，
FP32、TF32关闭、batch1、20图预热+同200图CUDA同步测量。
本轮P300的AP与T1逐个seed、逐个AP指标完全一致；延迟为本轮重新测量，
与T1旧记录27.625ms的微小差异不归因于模型变化。

![P1 accuracy, latency and crowded coverage](../../results/figures/v3_proposal_sweep.png)

### 覆盖率和类别否决条件

| 每级P | 最后一级全GT覆盖IoU0.5 | 拥挤图覆盖IoU0.5 | 拥挤图Δ覆盖（百分点） | 最差类别ΔAP |
|---|---:|---:|---:|---|
| 300 | 99.130% | 98.067% | +0.000 | 基准 |
| 200 | 98.660% | 97.072% | −0.995 | Anthracnose Fruit Rot −0.739 |
| 150 | 98.260% | 96.396% | −1.670 | Angular Leafspot −0.849 |
| 100 | 97.327% | 94.388% | −3.679 | Powdery Mildew Fruit −1.692 |

全验证集1775个GT，拥挤图（每图≥5个目标）含592个GT。覆盖率是NMS前每个GT的
最佳类别无关IoU，允许一个框覆盖多个GT，**不是COCO AR**。
各类别平均AP下降均未超过2.0，但P150/P100拥挤图覆盖率下降超过1个百分点。
早期头的覆盖差距在后续头中没有完全恢复；“目标不多”并不能证明小P足够。

### 判定与后续方向

**P1三个候选全部未通过，保持P300。** P200的实际平均ΔAP=−0.303468，略低于−0.3；
同时AP75下降0.538、p50降幅远低于10%，不因AP接近边界而放宽门槛。
P150/P100的AP、AP75、速度和拥挤覆盖率门槛均未满足。所有候选的速度改进都不足1%，
不能将其表述为实用加速，更不能宣称通过G4的1.4倍目标。

这否定的是当前冻结H4权重、RTX3090、FP32协议下的直接固定减框方案，
不等于证明小P重训或其他部署后端无效。未做分段耗时分析，不能由这张表直接判定
瓶颈来自CPU、backbone或某个算子。逐级剪框在投入训练前应先证实实际可节省的延迟。
下一优先级仍为同预算H6对照与独立质量监督；不继续用开发集搜索更多固定P值。

### 审计和复现

- 36行评测、1296行逐级/逐类覆盖率完整，无重复或非有限AP。
- 九组P300与T1所有AP指标最大偏差0；三份权重各两张真实图的注入噪声推理逐bit一致。
- 三份checkpoint、源码快照和原始/评测标注核验通过；覆盖GT分母与原始标注一致。
- 73项自动测试通过；主进程exit code 0、`COMPLETE.json`存在，GPU1结束后0 MiB。
- 未覆盖历史权重；运行与预测保存在`runs/v3/proposal_sweep_20261013/`。

产物：
[36次评测](../../results/raw/v3_proposal_sweep.csv) ·
[配对汇总](../../results/raw/v3_proposal_sweep_paired.csv) ·
[总体汇总](../../results/raw/v3_proposal_sweep_summary.csv) ·
[逐类AP](../../results/raw/v3_proposal_sweep_per_class.csv) ·
[逐级覆盖率](../../results/raw/v3_proposal_sweep_coverage.csv) ·
[Gate与审计](../../results/raw/v3_proposal_sweep_gate.json) ·
[manifest](../../results/raw/v3_proposal_sweep_manifest.json) ·
[配置](../../results/raw/v3_proposal_sweep_config.yaml) ·
[逐bit对拍](../../results/raw/v3_proposal_sweep_parity.json)。

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_summarize_proposals.py \
  runs/v3/proposal_sweep_20261013
```
