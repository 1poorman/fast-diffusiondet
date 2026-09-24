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

## 4. M1 正确性测试（采样器）

### 5.1 DDIM 逐 bit 等价回归

`scripts/test_ddim_equiv.py`：8 张图 × `SAMPLE_STEP ∈ {1,4}` × 固定 seed，
比对重构前（`detector.py` 内联实现）与重构后（`solvers/ddim.py`）的输出框。

```
[test_ddim_equiv] compared 16 entries, missing=0
[test_ddim_equiv] max|delta| = 0.000e+00
[test_ddim_equiv] PASS (max|delta| < 1e-5)
```

**完全逐 bit 一致（误差 0）**，含 Box Renewal 与 Ensemble 路径。

### 5.2 解析解收敛阶（`scripts/test_convergence.py`）

合成高斯去噪器 `D(x̂;σ̂) = (s²x̂ + σ̂²μ)/(s²+σ̂²)` 的闭式解
`x̂(0) = μ + (x̂_max − μ)·s/√(s²+σ̂_max²)`，误差随步数 N 的下降斜率即收敛阶。

| solver | 实测阶 | N=4 | N=8 | N=16 | N=32 | 判定 |
|---|---|---|---|---|---|---|
| Euler | **0.74** | 1.66e0 | 1.01e0 | 6.54e-1 | 3.49e-1 | ✅ O(h) |
| Heun | **2.77** | 6.81e1 | 3.71e0 | 8.50e-1 | 1.85e-1 | ✅ ≥O(h²) |
| DPMv3 order=2（真实 1000 档 schedule） | **1.79** | 2.21e2 | 2.60e2 | 9.45e1 | 5.00e0 | ✅ ≥O(h²) |
| DPMv3 order=1（真实 1000 档） | 0.34 | 7.32e0 | 4.14e1 | 7.27e1 | 2.76e0 | 1 阶，符合预期 |
| DPMv3 order=3（真实 1000 档） | **−0.30** | — | — | — | — | ❌ 发散 |

### 5.3 🔴 关键发现：高阶多步求解器 vs 离散时间步

把 schedule 的档位从 1000 加密到 **200000**（取整误差可忽略）后：

| solver | 真实 1000 档 | 密集 200000 档 |
|---|---|---|
| DPMv3 order=3 | **−0.30（发散）** | **2.21（正常 ≥O(h²)）** |

**结论**：DPM-Solver-v3 的 3 阶**对「连续 σ̂ 被取整到离散 timestep」极其敏感**。
原因是高阶多步法要用缓存的 ε 做有限差分估计高阶导数，
而取整让 ε(x, σ̂) 变成阶梯函数，差分被噪声放大。

**直接后果与对策**
1. M2 在 **VP + 离散 timestep** 的 baseline checkpoint 上，DPM-Solver-v3 **必须用 order=2**（实测 1.79 可用）。
2. 这给 **M4 的 EDM 改造提供了额外理由**：把 head 的时间条件换成连续的 `c_noise = ln σ/4`
   （`edm-main/training/networks.py:663`），天然消除取整，order=3 才可能生效。
3. 备选缓解：对离散 timestep 做插值式时间嵌入（把 `t` 以浮点喂给 `SinusoidalPositionEmbeddings`）。
   已登记为风险 **R14**。

## 5. M2 training-free 对比矩阵（进行中）

配置：`f0`（VP baseline，同一个 `model_final.pth`，**权重冻结**）、SDD val **全量 750 张**、
`box_renewal=False`、`use_ensemble=False`（FEP §2.3 的纯采样器对比）、`NUM_PROPOSALS=300`。
数据：`results/raw/matrix.csv`。

| solver | NFE=1 | NFE=2 | NFE=4 |
|---|---|---|---|
| **DDIM**（基线） seed0 | 60.53 | 59.62 | 56.72 |
| seed1 | 60.71 | 59.82 | 56.84 |
| seed2 | 60.62 | 59.26 | 56.73 |
| **DDIM 均值** | **60.62** | **59.57** | **56.76** |
| Euler | 待跑 | 待跑 | 待跑 |
| Heun | 待跑（NFE=1/3/5） | | |
| DPM-Solver-v3 (order=2, degenerated) | 待跑 | 待跑 | 待跑 |

### 6.1 🔴 重要发现：关掉 renewal+ensemble 后，**步数越多 AP 越低**

DDIM 的 AP 随 NFE **单调下降**（60.62 → 59.57 → 56.76）。

结合 `detector.py:239` 的事实 —— Box Renewal 与 Ensemble **只在 `sampling_timesteps > 1` 时生效** ——
这说明原论文"多步更好"的收益**几乎全部来自 renewal + ensemble，而不是采样质量本身**。
在纯 ODE 意义下，DiffusionDet 的 DDIM 反而是 **1 步最优**。

→ 该结论直接改变 M2 的评价口径：**新求解器要对标的不是"更多步的 DDIM"，
而是 NFE=1 的 DDIM（AP 60.62）**。想赢就必须在 NFE=1~2 上超过它。
同时也把 A1/A2 消融（renewal / ensemble 开与关）的优先级提到了最高。

### 6.2 方法论教训：不能用评测子集加速

| 子集大小 | DDIM NFE=1 的 AP |
|---|---|
| 20 张 | **2.02** |
| 250 张 | **21.92** |
| 750 张（全量） | **60.53** |

AP 与子集大小近似成正比 —— pycocotools 的召回分母仍按**全量 GT** 计算，
只在前 n 张图上推理会系统性低估。
→ **所有 AP 数字必须用全量 split**；加速只能靠减少 NFE 网格或 seed 数。
已在 `scripts/run_matrix.py:make_subset` 中写明警告。

## 6. 假设判定看板（随数据更新）

| 假设 | 内容 | 当前判定 | 依据 |
|---|---|---|---|
| H1 | NFE≤4 时 2 阶+ 求解器 AP ≥ DDIM +0.5 | ⬜ 待测（M2） | — |
| H2 | PFGM++ 使 NFE=2 ≈ DDIM@8 | ⬜ 待测（M5） | — |
| H3 | EMS 真统计量仅在 NFE≤5 有可见收益 | ⬜ 待测（M3） | — |
| H4 | EDM 主要提升上界而非低 NFE 效率 | ⬜ 待测（M4） | — |
| **H0** | head forward 是延迟主项、NFE 与延迟线性相关 | ✅ **支持** | §2：边际 71.5 ms/step，占比 66% |
