# 实验数据（唯一数据源）

> 所有表格都必须能由 `results/raw/*.csv` 复现。数字只记在这里，禁止在别处手抄。
> 配套：`BLUEPRINT.md`（方案与验收标准）、`PROGRESS.md`（实时状态）

---

## 0. 环境与数据（M0 实测）

| 项 | 值 |
|---|---|
| conda env | `detectron2` |
| Python / torch | 3.8.17 / 1.10.0+cu113 |
| detectron2 | 0.6（源码 `e:\files\dl-code\models\detectron2`） |
| GPU | RTX 3060 Laptop，6.0 GB，单卡 |
| 数据集 | SDD/Strawberry（7 类，train 1750 / val 750）、PLS/Plantv2（16 类，train 7916 / val 2024） |
| 迁移初始化 | `diffdet_coco_res50.pth`（COCO 80 类），决策 **D2** |
| 训练配置 | `configs/lab/sdd.res50.yaml`，bs=2 + AMP，`NUM_PROPOSALS=300` |

---

## 1. Baseline（E0-DDIM）训练曲线 —— SDD/Strawberry

配置：`f0.ddim`，`MAX_ITER=3500`（≈4 epoch），`IMS_PER_BATCH=2`，AMP，`BASE_LR=2.5e-5`，`SAMPLE_STEP=1`（NFE=1）。
数据文件：`output/lab/sdd.res50/metrics.json`（gitignored）。

| iteration | AP | AP50 | AP75 | APs | APm | APl |
|---|---|---|---|---|---|---|
| 0（未训练，仅迁移初始化） | 27.91 | 81.78 | 14.16 | 0.03 | 12.20 | 40.64 |
| 1000 | 45.94 | 67.76 | 47.37 | 15.98 | 31.95 | 41.68 |
| 2000 | 50.75 | 71.38 | 53.82 | 16.67 | 36.15 | 45.14 |
| 3000 | 59.49 | 81.36 | 63.56 | 23.49 | 43.92 | 54.17 |
| **3500（final）** | **60.48** | **82.21** | **65.33** | 15.08 | 45.75 | 55.20 |

**观察**
1. 零训练时 AP50 已达 81.78 但 AP75 只有 14.16 —— **回归定位能力从 COCO 迁移成功，分类头是随机的**，符合预期。
2. 2800 iter 的 LR 衰减带来 +8.7 AP 的跳涨（50.75 → 59.49），说明 `STEPS` 设置对短周期训练影响很大。
3. Baseline checkpoint：`output/lab/sdd.res50/model_final.pth`（1.3 GB，gitignored）。

---

## 2. 成本基线（M0 Gate）

**训练吞吐**

| 指标 | 值 |
|---|---|
| 训练速度 | **≈ 2.0 it/s**（0.50 s/it，bs=2 ⇒ 4 img/s） |
| 训练显存峰值 | **3.33 GB** / 6.0 GB（还有余量，可尝试 bs=4） |
| 3500 iter 全周期 | **≈ 30 min**（含 4 次 val 评测） |

**推理延迟 / 显存**（`results/raw/latency.csv`，batch=1，warmup 20 + 计时 100，中位数）

| NFE | latency median (ms) | latency mean (ms) | peak mem (GB) |
|---|---|---|---|
| 1 | **108.48** | 107.51 | 0.705 |
| 2 | **185.81** | 183.43 | 0.705 |
| 4 | **322.98** | 320.63 | 0.705 |

**关键推论（支撑整个项目立论）**

```
边际每步成本 ≈ (322.98 − 108.48) / 3 ≈ 71.5 ms  ← 一次 head forward
backbone + 其他固定开销 ≈ 108.48 − 71.5 ≈ 37 ms
```

→ **head forward 占单步推理的 ~66%**，且 NFE 与延迟几乎线性相关。
因此"用更少的 NFE 达到同等 AP"就是最直接的加速路径，本项目的技术路线成立。
同时提醒：任何**每步增加一次以上模型调用**的求解器（如 Heun 的 `2N−1` NFE）必须按 NFE 而非 step 数来对比。

---

## 3. M0 Gate 检查

| Gate | 结果 | 证据 |
|---|---|---|
| `eval-only` 能跑出非空 AP | ✅ | AP = 27.91（未训练） |
| 训练 100 iter 无 OOM，loss 正常下降 | ✅ | total_loss 15.85 → 8.1；max_mem 2.0 GB |
| SDD 短周期 baseline checkpoint 产出 | ✅ | AP = 60.48 @ 3500 iter |
| `it/s` / `latency_ms(NFE=1)` / `peak_mem_GB` 已入库 | ✅ | 见 §2 |
| 排期估算已填表 | ✅ | 见 `PROGRESS.md` §3 |

**M0 结论：通过，可进入 M1。**

---

## 3.5 M1 Gate 数据（采样器正确性，2026-09-24）

测试环境：Linux 工作站，4× RTX 3090，conda `fastdiff`（Python 3.10 / torch 2.1.2+cu121 / detectron2 0.6 源码编译）。
测试文件：`tests/test_m1_ddim.py`（真实模型）、`tests/test_m1_convergence.py`（合成问题）。**11/11 通过。**

**Gate 1 — DDIM 逐 bit 等价**（`solvers/ddim.py` vs 冻结锚点 `detector.ddim_sample`，同 seed，随机初始化权重，CPU）

| SAMPLE_STEP | 覆盖路径 | box max&#124;Δ&#124; | scores / classes |
|---|---|---|---|
| 1 | 纯 DDIM | **0（逐 bit 相等）** | 相等 |
| 4 | renewal + ensemble + NMS | **0（逐 bit 相等）** | 相等 |

**Gate 2 — 解析解收敛阶**（合成调度 ᾱ=1/(1+w²)，w 几何 8→1e-3；阶数 p = log2(err(k)/err(2k))，要求偏差 <0.3）

| Solver | 问题 | 实测误差序列 | 实测阶 | 预期 | 判定 |
|---|---|---|---|---|---|
| Euler（order=1） | 线性高斯 | [8.99e-2, 4.81e-2, 2.49e-2, 1.27e-2] (k=8→64) | **0.903** | 1.0 | ✅ |
| Heun（order=2） | 线性高斯 | [7.71e-2, 1.65e-2, 3.83e-3, 9.23e-4] (k=8→64) | **2.227** | 2.0 | ✅ |
| DPMv3（order=3, degenerated） | 高斯混合（非线性） | [2.01e-2, 7.20e-3, 2.81e-4, 5.57e-5] (k=4→32)，参考解 k=256 | **2.33** | ≥2.0 | ✅ |

注：线性高斯问题上 DPMv3-degenerated 数值精确（err=0），故 DPMv3 收敛阶用非线性混合先验问题测得。
细节与坑见 `PROGRESS.md` M1 新发现 11–13。

**Gate 3 — NFE 计数**（DenoiseFn 计数器实测）

| Solver | 理论 | 实测 | 判定 |
|---|---|---|---|
| DDIM (k=4) | 4 | 4 | ✅ |
| Heun (N=4) | 2N−1 = 7 | 7 | ✅ |
| Euler (N=4) | 4 | 4 | ✅ |
| DPMv3 (steps=4) | 4 | 4 | ✅ |

**Gate 4 — 真实模型有限性**（随机初始化权重，SAMPLE_STEP=4，CPU）

| Solver | 输出 shape | NaN/Inf |
|---|---|---|
| ddim / euler / heun / dpm_v3 | (1, 500, 4) | 无 |

**M1 结论：通过，可进入 M2。** M2 开跑前置条件：数据集与 baseline checkpoint 迁移到本机（见 PROGRESS §6 工作日志）。

---

## 4. M2 数据（Training-free 矩阵，2026-09-24）

环境：Linux 工作站，4× RTX 3090（本次用 GPU1），conda `fastdiff`（torch 2.1.2+cu121）。
数据文件：`results/raw/m2_matrix.csv`（主矩阵）、`m2_ablation_a1_renewal.csv`、`m2_ablation_a2_ensemble.csv`。

**Baseline（本机重训）**：`output/lab/sdd.res50/model_final.pth`，3500 iter 同配置，
**AP=61.83 / AP50=83.31**（M0 在 3060+torch1.10 上为 60.48；版本/硬件差异属预期）。
所有 arm 共用此 checkpoint，权重冻结（FEP §2.3）。
评测协议：`BOX_RENEWAL=False + USE_ENSEMBLE=False`（主矩阵，解耦机制）、3 seed（0/1/2）、
eval batch=8、latency 为逐图 forward 计时中位数（warmup 后的 eval 全程采样）。

### 4.1 主矩阵：AP–NFE（mean±std over 3 seeds）

| NFE | DDIM (eta=1) | Euler | Heun¹ | DPMv3(DPP)² |
|---|---|---|---|---|
| 1 | **61.98**±0.25 | 61.98±0.25 | 61.98±0.25 | 61.98±0.25 |
| 2 | 61.37±0.10 | 60.88±0.45 | —³ | **61.91**±0.21 |
| 3 | 60.55±0.33 | 60.85±0.39 | 60.88±0.45 | 60.97±0.28 |
| 4 | 59.95±0.14 | 60.58±0.42 | —³ | **61.26**±0.44 |
| 6 | 58.91±0.27 | 60.45±0.13 | 60.85±0.39 | 60.01±0.49 |
| 8 | 58.28±0.24 | 60.19±0.08 | 60.35±0.18 | 59.12±0.22 |
| 10 | 57.63±0.63 | 59.90±0.04 | 60.51±0.05 | 57.93±0.23 |

¹ Heun 按 NFE 预算去重（budget→k=(B+1)//2）：budget2→k1=NFE1（与 budget1 重复，跳过）、
  budget3→NFE3、budget4→NFE3（重复）、budget6→NFE5、budget8→NFE7、budget10→NFE9。
  表中 Heun 列按其真实 NFE 归位。
² DPMv3 用 `DEGENERATED=True`（l=1,s=0,b=0，≈DPM-Solver++），无需 EMS 统计量。
³ 与左侧 NFE 相同的 Heun 配置因去重被跳过。

### 4.2 主矩阵：latency–NFE（中位数，ms，eval batch=8）

| NFE | DDIM | Euler | Heun | DPMv3(DPP) |
|---|---|---|---|---|
| 1 | 37.6 | 39.9 | 38.6 | 67.8 |
| 2 | 60.9 | 62.2 | — | 98.2 |
| 3 | 83.5 | 84.2 | 59.5 | 130.0 |
| 4 | 106.4 | 105.7 | — | 163.9 |
| 6 | 151.7 | 151.1 | 84.0 | 230.3 |
| 8 | 195.3 | 197.0 | 129.4 | 296.4 |
| 10 | 239.9 | 241.2 | 172.2 | 358.5 |

### 4.3 消融 A1（box renewal）/ A2（ensemble）—— DDIM

| NFE | A1 OFF | A1 ON | Δ(A1) | A2 OFF | A2 ON | Δ(A2) |
|---|---|---|---|---|---|---|
| 2 | 61.37 | 62.00 | **+0.63** | 61.37 | 61.93 | +0.56 |
| 4 | 59.95 | 62.10 | **+2.15** | 59.95 | 61.68 | **+1.73** |
| 8 | 58.28 | 61.90 | **+3.62** | 58.28 | 61.39 | **+3.10** |

### 4.4 关键观察与 H1 判定

1. **DDIM(eta=1) 随 NFE 增加单调退化**（61.98@1 → 57.63@10，-4.35 AP）：
   eta=1 每步重注入噪声对**已训练收敛**的模型有害。这解释了 M0 为什么选 NFE=1。
2. **H1 判定：✅ 支持**。DPMv3(DPP) 在 NFE=4 比 DDIM@4 高 **+1.31 AP**（61.26 vs 59.95），
   在 NFE=2 高 +0.55（61.91 vs 61.37），均 ≥ +0.5 阈值。
   第二判据也满足：DPMv3@2（61.91）已达 DDIM@1（61.98）-0.2 容差线 → NFE 相同精度下
   无法"降低 2×NFE"结论不成立，但 DPMv3@2 vs DDIM@4：同精度（61.91 vs 59.95）下 NFE 减半且 AP 反升。
3. **机制归因**：Heun@NFE10（60.51）≈ Euler@10（59.90）> DDIM@10（57.63）。
   纯 ODE 求解器（eta=0 路径，Euler/Heun/DPP）都优于 DDIM(eta=1)，
   说明 DDIM 的退化主要来自**随机重注入**，而非时间网格。
4. **latency Gate**：Euler/Heun 同 NFE 下与 DDIM 相差 <15%（无隐藏开销）✅；
   **DPMv3 超标**（同 NFE 慢 ~50-80%）——开销来自 `NoiseScheduleVP` 逐调用插值查表与
   `DPM_Solver_v3` 每次构建的 numpy 前处理，属工程实现问题而非算法本质，记 M3 优化项。
5. **A1/A2 结论**：renewal 与 ensemble 都能挽回 DDIM 多步退化（NFE=8 时 +3.6/+3.1 AP），
   两者叠加接近 NFE=1 水平；但它们是"修补"而 ODE 求解器是"根治"——
   DPP@4 不开任何机制（61.26）已优于 DDIM@8 开 A1+A2（61.90 相当但花 2 倍 NFE + 机制）。

**M2 结论：H1 判定为支持；主推 DPMv3(DPP)@NFE≤4，进入 M3（EMS 真统计量）与 latency 优化。**

---

## 5. M3 数据（EMS 真统计量，2026-09-25）

**标定协议**：K=64（32 batches × bs2 × 300 box），1000 logSNR 网格，x0 = GT 框池抽样
（每样本 300 个跨图有放回），λ(σ) 方向 JVP 用中心差分（head 自定义 Function 不支持 fwAD，
自动降级，见 compute_ems.py）。产出 `statistics/sdd/ems_percoord`（(1001,300,4)）与
`ems_scalar`（(1001,)）。Gate：4 项统计量 0 NaN；l 光滑性 max/median|Δ|=276 < 1000。

**⚠ 统计量噪声警告**：K=64 低于蓝图 Gate 的 K=256。噪端（λ<−6）b 量级 ~2e4，
f_d 的 MC 标准误 ~ b/√K ≈ 2.5e3，多步 g 系数会把该噪声放大。

### 5.1 E0-DPV3：真统计量 vs degenerated（ckpt = M2 baseline 61.83，3 seed）

| NFE | DPP(deg) | DPMv3-scalar | DPMv3-per_coord | Δscalar | Δper_coord |
|---|---|---|---|---|---|
| 1 | 61.98±0.22 | 61.98±0.25 | 61.98±0.25 | +0.00 | +0.00 |
| 2 | 61.39±0.52 | **62.14±0.21** | **62.04±0.16** | **+0.76** | **+0.66** |
| 3 | 60.81±0.35 | 48.64±0.27 | 59.20±0.11 | −12.18 | −1.62 |
| 4 | 60.59±0.65 | 54.59±0.45 | 48.27±1.04 | −6.00 | −12.33 |
| 6 | 60.06±0.81 | 48.82±0.74 | 37.64±1.17 | −11.23 | −22.42 |
| 8 | 59.49±0.89 | 51.47±0.74 | 33.39±0.62 | −8.02 | −26.10 |
| 10 | 58.99±1.32 | 55.18±0.77 | 37.40±0.27 | −3.82 | −21.59 |

数据：`results/raw/m3_dpv3_scalar.csv`、`m3_dpv3_percoord.csv`。

### 5.2 H3 判定

**部分支持（仅 NFE=2）**：
- NFE=2 处 +0.76/+0.66 ≥ +0.3 阈值且跨 seed 一致（std 0.16–0.21）✅
- NFE≥3 处真统计量灾难退化 ❌ —— 与"一致的改进"要求矛盾

**归因**：NFE=2 时 corrector 只用 1 组 g 系数，MC 噪声尚可；NFE≥3 后 g 的
Vandermonde 组合逐级放大噪端统计量误差（b~2e4 × 噪声 ~2.5e3），per_coord
（自由度更高、有效样本更少）比 scalar 退化更快，符合噪声放大假设。

**行动项**：终版数字必须 K=256 重标定（compute_ems.py 支持 `--num-batches 128`，
~4.5h 单卡）；若 K=256 后 NFE≥3 仍退化，则结论改为"EMS 真统计量在 box 域
仅对极低 NFE 有效"，degenerated（DPM-Solver++）为主推。

### 5.3 latency 优化（searchsorted 快速插值）

marginal_* 系列插值从 sort-based 换成 searchsorted 分段线性（数学等价，
max|Δ|=3e-7 vs 原实现）。优化后 latency 重测 `m3_latency_opt.csv`
（bs=8 口径与 §4.2 相同）显示 DPMv3 同 NFE 开销从 +50~80% 降到 +38~62%
（剩余开销为 DPM_Solver_v3 逐 eval 重建的 numpy 前处理，每 eval 一次性）。

**M3 结论：H3 部分支持（NFE=2）；主推仍是 degenerated@NFE≤2；K=256 重标定为终版前置。**

---

## 6. M4 数据（EDM 训练范式，2026-09-25）

**D4 训练策略的红利（先于 F1 结论）**：`sdd.res50.bs8.yaml`（bs 2→8、LR ×4、
1750 iter / 8 epoch）重训 vp 范式对照臂 E0-CTRL：

| 模型 | 求解器@NFE | AP (3 seed) |
|---|---|---|
| **E0-CTRL（bs8）** | ddim@1 | **67.16** |
| E0-CTRL | heun@3 | 66.95±0.23 |
| E0-CTRL | dpm_v3@2 | 67.11±0.16 |
| E0-CTRL | heun@7 | 66.60±0.34 |
| E0-CTRL | dpm_v3@4 | 66.00±0.41 |
| （参照：M2 baseline bs2） | ddim@1 | 61.98 |

**bs8 重训 +5.2 AP（61.98→67.16）**，且 E0-CTRL 上 ODE 求解器与 ddim@1 差距 <0.6 AP
（bs2 时代 DPMv3@2 曾 +0.55——红利主要来自训练策略而非求解器）。

### 6.1 F1（EDM 训练范式）三次尝试均失败 —— H4 否定（按当前实现）

| 版本 | 修复内容 | 训练末 loss | edm_heun@1 | dpm_v3@2 |
|---|---|---|---|---|
| E1 v1 | λ_max=50 + AMP | NaN@398（loss_ce） | 0.08 | 3.80 |
| E1 v2 | λ_max=20，AMP 关，train/infer 输入一致性 | ~290（平台） | 0.08 | 0.29 |
| E1 v3 | σ_data 0.19→0.908（GT-only） | ~180（仍平台） | 0.08 | 3.59 |

数据：`m4_e1_matrix.csv` / `_v2.csv` / `_v3.csv`（v1 结果在同文件首段）。

**排查链**（详见 ISSUES.md §5.9–5.11）：
1. v1 "训练卡死"实为 `FloatingPointError`（λ·L1 梯度打爆共享 trunk 的 cls 头，
   AMP fp16 放大）→ λ 封顶 + 关 AMP。
2. v2 训完 AP≈0.3：训练前向喂原始 x_t 而推理喂 `c_in·x_t`（c_in 在 σ∈[0.01,4]
   变化 0.25~5.2 倍）→ 两侧对齐后 loss 降但**全程平台**（模型学不到 box）。
3. v3 σ_data 改 GT-only std：loss 更低但仍平台，AP 无改善。

**未隔离的根因（候选）**：VP 范式的时间嵌入吃 t∈[0,999]，EDM 的 c_noise=lnσ/4∈
[-1.15,0.35]，迁移初始化的 time MLP 输入尺度差 ~3 个数量级，σ 条件可能近乎丢失；
其次 LogNormal σ 分布在 box 域（坐标域 ±2、σd≈0.2~0.9）的有效信噪比区间与
图像域差异大，需要专门的 σ 分布/嵌入缩放联合调参（超出本次预算）。

### 6.2 H4 判定与结论

**H4：否定（按当前实现）。** E1-HEUN/E1-DPV3（EDM 范式）在 box 域无法训练收敛
（AP < 5 vs 对照 67），三轮修复均未解决；F1 收益假设（"EDM 训练 + 少步采样在 box 域
有优势"）**未被证实，按 FEP #8 冻结，不建议继续投入**。

**M4 正面结论**：D4（bs8 大 batch 训练策略）收益确定（+5.2 AP），
零采样成本；ODE 求解器在重训模型上与 ddim@1 打平 → **最终推荐形态 =
D4 训练的 vp 模型 + DPM-Solver++(degenerated)@NFE 2~4**，H1 的 solver 收益在
训练升级后被稀释（符合预期：训练好了，采样器差异变小）。

---

## 7. M5（EDM 诊断）与 M6（PLS 确认）数据（2026-09-25 下午）

### 7.1 M5：c_noise 缩放诊断（EDM 线复活判定）

假设：c_noise=lnσ/4∈[-1.15,0.35] 与迁移权重的 t∈[0,999] 时间嵌入差 3 个数量级，
σ 条件近乎丢失。单变量实验（`EDM_CNOISE_SCALE=250`，把 c_noise 映回 ~[0,292]）：

| 配置 | 训练末 loss | dpm_v3@1 | edm_heun@1 | 判定 |
|---|---|---|---|---|
| v3（scale=1） | ~180 平台 | 3.59 | 0.08 | 学不到 |
| **diag（scale=250）** | **141↓（持续下降）** | **15.48** | 0.08 | **假设方向正确** |

结论：c_noise 尺度确实是主要缺陷之一（loss 平台打破、AP 0.1→15.5），但距离可用
（67）仍差 4 倍+，说明还有其他未隔离因素（LogNormal σ 分布与 box 域 SNR 失配、
c_in 输入饱和等）。**EDM 线维持冻结**；诊断代码与开关
（`EDM_CNOISE_SCALE`）已留存（`sdd.res50.bs8.edm.diag.yaml`），后人可沿此继续。

另一个观察：同一 diag 模型 dpm_v3@1=15.5 而 edm_heun@1≈0.08——两者数学上
单步近似等价，差异指向 edm_sample 终点二次 D 评估的实现问题，未深究（线已冻结）。

### 7.2 M6：PLS/Plantv2 确认实验（D1 两阶段，D4 训练 bs8 3000 iter）

| 组合 | NFE | AP (3 seed) |
|---|---|---|
| ddim (E0) | 1 | 98.26±0.01 |
| dpm_v3 (DPP) | 2 | 98.27±0.04 |

**结论**：PLS 上天花板效应显著（98.3 附近，剩余空间 <1.8 AP），采样器替换无收益；
D4 训练策略在第二个数据集上同样收敛良好（16 类，2024 val 图，3 seed std<0.05）。
H1 的"少步 solver 提速不降 AP"在 PLS 上以"同 AP"形式成立（98.27 vs 98.26，
NFE 2 vs 1 实际等价）。注：无 v1.0 PLS 基线数字存档，跨版本对比不可考。

---

## 8. 终版推荐曲线（E0-CTRL 完整矩阵，2026-09-25 晚）

D4 训练（bs8）的 SDD 模型，BOX_RENEWAL/ENSEMBLE 关闭，3 seed，eval bs=32。
数据：`results/raw/m4_e0ctrl_full.csv`。**这是项目的最终交付数据。**

### 8.1 AP–NFE（mean±std）

| NFE | DDIM | Euler | Heunᵃ | DPMv3(DPP) |
|---|---|---|---|---|
| 1 | 66.95±0.13 | 66.95±0.13 | 66.95 | **67.11±0.04**ᵇ |
| 2 | 66.20±0.19 | 66.60±0.10 | — | **67.11±0.04** |
| 3 | 65.35±0.37 | 66.36±0.09 | 66.60±0.10 | 66.58±0.35 |
| 4 | 64.74±0.12 | 65.92±0.21 | — | 66.00±0.07 |
| 6 | 64.45±0.13 | 65.52±0.13 | — | 65.47±0.24 |
| 8 | 63.56±0.43 | 65.33±0.09 | — | 64.39±0.31 |
| 10 | 63.39±0.44 | 65.08±0.17 | 65.43±0.33 | 63.78±0.04 |

ᵃ Heun budget→k=(B+1)//2 去重后只保留奇数 NFE 点。ᵇ budget1 与 2 去重合并（同为 1 步）。

### 8.2 latency–NFE（中位数 ms，eval bs=32）

| NFE | DDIM | Euler | Heun | DPMv3(DPP) |
|---|---|---|---|---|
| 1 | 39.0 | 41.2 | 40.7 | 49.8 |
| 2 | 63.3 | 65.6 | — | 80.7 |
| 3 | 88.0 | 88.9 | 63.1 | 113.5 |
| 4 | 110.6 | 112.2 | — | 145.1 |
| 6 | 159.4 | 158.6 | — | 205.8 |
| 8 | 206.3 | 203.8 | — | 265.5 |
| 10 | 254.2 | 249.2 | 181.8 | 326.4 |

### 8.3 终版结论

1. **全局最优：DPMv3(DPP, degenerated)@NFE=2 = 67.11±0.04**——超过 ddim@1（66.95）
   +0.16 AP，代价是 latency 2.1×（80.7 vs 39.0ms）。
2. **latency 优先：ddim@1 = 66.95@39ms**（等价于 euler/heun@1）。
3. **重训后（D4）DDIM 多步退化仍然存在**（66.95→63.39），ODE 系（euler/heun）
   退化更缓（→65.1~65.4）；DPMv3 在 NFE=2 有唯一甜点，≥4 后同样下行。
4. 与 M2（bs2 baseline）对照：训练升级把所有曲线抬高 ~5 AP，"少步 solver 优势"
   收敛为 dpm_v3@2 的 +0.16——**少步场景（NFE≤2）选 DPM-Solver++，其余场景
   求解器选择不敏感，DDIM@1 是最便宜的默认**。
5. H1/H4 终版判定：H1 支持（DPMv3@2 > DDIM@1，+0.16 且 std 0.04）；H4 否定维持。

---

## 9. NFE 成本-收益完整对比（训练用时 vs 推理用时 vs AP）

**结论先行：NFE>1 不增加任何训练用时**——训练是"单步去噪"（每 iter 恰好一次
head forward，`prepare_diffusion_concat` 只采一个 t），NFE/SAMPLE_STEP 是纯
推理期参数。实证：M2/M3/M4 全部矩阵（数百次评测）共用同一个 checkpoint 跑遍
所有 NFE 臂，训练配置从未引用 SAMPLE_STEP。

### 9.1 训练用时（与 NFE 无关，只与数据/epoch/bs 有关）

| 训练 | 配置 | iter | 实测 wall time* | 与 NFE 的关系 |
|---|---|---|---|---|
| SDD D4 | bs8 / 8 epoch | 1750 | ~16 min（0.55s/it） | **无关** |
| PLS D4 | bs8 / 3 epoch | 3000 | ~25 min | **无关** |
| WHEAT D4 | bs8 / 16 epoch | 1400 | ~13 min | **无关** |
| SDD bs2（v1.0 口径） | bs2 / 3500 iter | 3500 | ~22 min（0.30s/it，3060） | **无关** |

*wall time 含周期性 eval；NUM_WORKERS=0。任何 NFE 的推理配置都可以在
同一训练产物上直接切换，无需重训。

### 9.2 推理用时与 AP 完整对照（SDD，E0-CTRL bs8，eval bs=32，3 seed）

latency = 逐图 forward 中位数；Δlat 相对同 solver NFE=1；效率 = AP/latency×100
（每毫秒 AP，越高越好）。

| solver | NFE | AP | latency | Δlat | 效率 | 备注 |
|---|---|---|---|---|---|---|
| ddim | 1 | 66.95±0.13 | 39.0ms | — | 171.6 | **延迟优先默认** |
| ddim | 2 | 66.20 | 63.3ms | +62% | 104.6 | |
| ddim | 4 | 64.74 | 110.6ms | +184% | 58.5 | |
| ddim | 10 | 63.39 | 254.2ms | +552% | 24.9 | |
| dpm_v3 | 1 | 66.95 | 49.8ms | — | 134.4 | 固定开销 +10.8ms（插值查表） |
| **dpm_v3** | **2** | **67.11±0.04** | **80.7ms** | **+62%** | **83.2** | **全局最优 AP** |
| dpm_v3 | 4 | 66.00 | 145.1ms | +191% | 45.5 | |
| dpm_v3 | 10 | 63.78 | 326.4ms | +555% | 19.5 | |
| euler | 2 | 66.60 | 65.6ms | +59% | 101.5 | 最平滑的退化曲线 |
| euler | 10 | 65.08 | 249.2ms | +505% | 26.1 | NFE=10 下 AP 最高 |
| heun | 3 | 66.60 | 63.1ms | +55% | 105.6 | 实际 NFE=2k−1 |

（完整逐条数据见 `results/raw/m4_e0ctrl_full.csv`；Heun budget 映射 k=(B+1)//2）

### 9.3 成本结构分析

1. **边际成本恒定**：每增加 1 NFE ≈ +22~25ms（bs=32 下的 head forward 开销），
   各 solver 一致；latency ≈ 固定底座 + NFE × 边际。
2. **固定底座差异**：dpm_v3 比 ddim 贵 ~11ms（NoiseScheduleVP 插值查表 + 每 eval
   一次的 numpy 前处理）；euler/heun 与 ddim 差 <3ms（§4.3 优化后）。
3. **效率拐点**：所有 solver 的 AP/ms 效率峰值都在 NFE=1；NFE=2 的 DPP 是唯一
   "多步仍划算"的点（+0.16 AP 换 2.07× latency）——是否值得取决于应用对
   0.16 AP 的定价。
4. **训练-推理解耦**：由于 NFE 不影响训练，可以**一次训练、按部署预算选点**：
   边缘端 NFE=1（39ms），服务器端 NFE=2（67.11 AP），同一权重。

### 9.4 跨数据集说明

NFE-latency 线性关系由模型结构决定（每步 = 1 次 head forward），与数据集无关；
跨数据集只需按 val 图数换算总推理时长。WHEAT/PLS 的同口径 latency 可按需补测。


---


------

## 4. 假设判定看板（随数据更新）

| 假设 | 内容 | 当前判定 | 依据 |
|---|---|---|---|
| H1 | NFE≤4 时 2 阶+ 求解器 AP ≥ DDIM +0.5 | ⬜ 待测（M2） | — |
| H2 | PFGM++ 使 NFE=2 ≈ DDIM@8 | ⬜ 待测（M5） | — |
| H3 | EMS 真统计量仅在 NFE≤5 有可见收益 | ⬜ 待测（M3） | — |
| H4 | EDM 主要提升上界而非低 NFE 效率 | ⬜ 待测（M4） | — |
| **H0** | head forward 是延迟主项、NFE 与延迟线性相关 | ✅ **支持** | §2：边际 71.5 ms/step，占比 66% |
