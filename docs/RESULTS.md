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

## 4. 假设判定看板（随数据更新）

| 假设 | 内容 | 当前判定 | 依据 |
|---|---|---|---|
| H1 | NFE≤4 时 2 阶+ 求解器 AP ≥ DDIM +0.5 | ⬜ 待测（M2） | — |
| H2 | PFGM++ 使 NFE=2 ≈ DDIM@8 | ⬜ 待测（M5） | — |
| H3 | EMS 真统计量仅在 NFE≤5 有可见收益 | ⬜ 待测（M3） | — |
| H4 | EDM 主要提升上界而非低 NFE 效率 | ⬜ 待测（M4） | — |
| **H0** | head forward 是延迟主项、NFE 与延迟线性相关 | ✅ **支持** | §2：边际 71.5 ms/step，占比 66% |
