# 工作状态实时记录

> 配套文档：`BLUEPRINT.md`（蓝图/方案/验收标准）、`RESULTS.md`（实验数据，M2 起）
> 更新规则：**每次动手前先加一行"进行中"，完成后改状态并勾 Gate。数字变化先落 `RESULTS.md`。**

最后更新：**2026-09-23** ｜ 当前里程碑：**M0（未开始）**

---

## 1. 里程碑总览

| 里程碑 | 状态 | 目标摘要 | 关键 Gate |
|---|---|---|---|
| **M0** 环境与基线建立 | ⬜ 未开始 | 接入 PLS/SDD、清理旧 bug、跑通单卡训练 | 短周期训练出非空 AP；测得 `it/s` |
| **M1** 采样器接口与正确性 | ⬜ 未开始 | `solvers/` 包：DDIM/Heun/DPMv3 | DDIM 逐 bit 一致；解析解收敛阶达标 |
| **M2** Training-free 矩阵 | ⬜ 未开始 | E0-{DDIM,HEUN,EULER,DPP} × NFE | H1：NFE≤4 下 ≥ +0.5 AP 或 2× 提速 |
| **M3** EMS 标定 + DPMv3 全功率 | ⬜ 未开始 | 有限差分 JVP 版 EMS | H3：EMS 相对 degenerated ≥ +0.3 AP |
| **M4** EDM 训练范式 | ⬜ 未开始 | 预条件 + 对数正态 σ + λ(σ) | H4：高 NFE 上界 ≥ +0.5 AP |
| **M5** PFGM++ 训练范式 | ⬜ 未开始 | Beta′ 重尾扰动 + 先验，扫 D | H2：NFE=2 ≈ EDM@NFE=8 |
| **M6** 组合/消融/报告 | ⬜ 未开始 | SDD→PLS 双阶段确认 | PLS 结论方向一致；H1–H4 全部判定 |

图例：⬜ 未开始 ｜ 🟨 进行中 ｜ ✅ 完成 ｜ ⛔ 阻塞 ｜ ➖ 放弃

---

## 2. 环境事实（2026-09-23 实测，已锁定）

| 项 | 值 |
|---|---|
| conda env | `detectron2` @ `D:\APP\conda\envs\detectron2` |
| Python / torch | 3.8.17 / **1.10.0+cu113** |
| detectron2 | **0.6**（源码 `e:\files\dl-code\models\detectron2`） |
| GPU | RTX 3060 Laptop **6.0 GB**，单卡 |
| numpy | 1.24.4 |
| PLS/Plantv2 | train 7916/7915，val 2024/2024，**16 类**（另有带标注 test2017 111 张） |
| SDD/Strawberry | train 1750/3932，val 750/1775，**7 类** |

**已确认的环境约束**
- ⚠ torch 1.10 **没有** `torch.autograd.forward_ad` → EMS 必须用**有限差分 JVP**（影响 M3）
- ⚠ 两个 json 的 `categories` 含 `(0,'_background_')` 但 GT 从不使用 → 必须自定义 id_map，`NUM_CLASSES`=16/7
- ⚠ 6GB 显存 → `IMS_PER_BATCH` 预计只能到 2，`NUM_WORKERS` 先设 0

---

## 3. 排期估算

> ⛔ **待 M0 实测 `it/s` 后填充**。先用 SDD 短周期（~3500 iter）估算单 arm 成本，
> 再据此决定阶段 A / 阶段 B 的 `MAX_ITER`。

| 里程碑 | 预估 GPU 时长 | 备注 |
|---|---|---|
| M0 | 待定 | 首次短周期训练 |
| M1 | 待定 | 主要是 CPU/单人 correctness 工作 |
| M2 | 待定 | 纯推理，4×8×3 次 eval |
| M3 | 待定 | EMS 计算 + 重跑 eval |
| M4/M5 | 待定 | 训练 × N arms，最大开销项 |
| M6 | 待定 | 仅 top-2 在 PLS 上完整训练 |

---

## 4. Gate 检查清单

### M0 Gate
- [ ] `eval-only` 能跑出非空 AP
- [ ] 训练 100 iter 无 OOM，loss 正常下降
- [ ] SDD短周期 baseline checkpoint 产出（AP > 0）
- [ ] `it/s` / `latency_ms(NFE=1)` / `peak_mem_GB` 三项已入 `RESULTS.md`
- [ ] 排期估算已填表

### M1 Gate
- [ ] DDIM 等价回归 `max|Δ| < 1e-5`
- [ ] 解析解收敛阶达标（Euler O(h)、Heun O(h²)、DPMv3 ≥ O(h²)）
- [ ] NFE 计数与理论值一致
- [ ] 三 solver 在真实 ckpt 上无 NaN

### M2 Gate
- [ ] AP–NFE / latency–NFE 曲线 ×4 入表（含 ±std）
- [ ] **H1 判定达成或明确否定**
- [ ] 同 NFE 下 latency 差异 < 15%
- [ ] A1(renewal) / A2(ensemble) 消融完成

### M3 Gate
- [ ] EMS 计算 < 2 小时（1000 网格，K=256）
- [ ] `l/s/b` 无 NaN，曲线光滑
- [ ] **H3 判定达成或明确否定**
- [ ] A8 消融完成

### M4 Gate
- [ ] 等预算下 Nf=10 的 AP ≥ 基线 −0.5
- [ ] `σ_data / P_mean / P_std` 标定入库
- [ ] A4 / A5 / A6 消融完成
- [ ] **H4 判定达成或明确否定**

### M5 Gate
- [ ] loss 稳定，无 NaN
- [ ] **H2 判定达成或明确否定**
- [ ] A7 粒度消融完成
- [ ] 单 arm 训练开销 ≤ EDM 版 1.1×

### M6 Gate
- [ ] PLS 与 SDD 结论方向一致（否则书面讨论）
- [ ] H1–H4 全部有明确判定
- [ ] Pareto 前沿表产出
- [ ] `RESULTS.md` 每张表可由 `results/raw/*.csv` 复现

---

## 5. 假设判定看板（随数据更新）

| 假设 | 内容 | 当前判定 | 依据 |
|---|---|---|---|
| H1 | NFE≤4 时 2 阶+ 求解器 AP ≥ DDIM +0.5 | ⬜ 待测 | — |
| H2 | PFGM++ 使 NFE=2 ≈ DDIM@8 | ⬜ 待测 | — |
| H3 | EMS 真统计量仅在 NFE≤5 有可見收益 | ⬜ 待测 | — |
| H4 | EDM 主要提升上界而非低 NFE 效率 | ⬜ 待测 | — |

---

## 6. 工作日志

### 2026-09-23

| 时间 | 事项 | 状态 |
|---|---|---|
| — | 勘查 `fast-diffusiondet` 结构、DDIM 接口、detectron2 耦合方式 | ✅ |
| — | 勘查 `DPM-Solver-v3-main` 求解器 API 与 EMS 统计量依赖 | ✅ |
| — | 勘查 `edm-main` 采样器 / 预条件 / 损失 / σ 分布 | ✅ |
| — | 勘查 `pfgmpp-main` 扰动核与重尾先验（**确认它是 EDM 的 fork，无散度/BFSD**） | ✅ |
| — | 实测 conda `detectron2` 环境：Py3.8.17 / torch1.10+cu113 / d2 0.6 / RTX3060 6GB | ✅ |
| — | 实测 PLS / SDD 数据集规模、类别、id 编号陷阱 | ✅ |
| — | 产出 `docs/BLUEPRINT.md` v1.0 | ✅ |
| — | 建立 `docs/PROGRESS.md` 反馈回环 | ✅ |

**本轮关键发现（影响后续设计）**
1. `fast-diffusiondet` 是二改版：`train.py` 硬编码 WGISD 路径、`detector_dpm3.py` 是**跑不起来的半成品**
   （缺 `multistep_predictor_update`、`forward` 调了不存在的 `ddim_sample`、`self.noise_schedule` 未赋值）。
   → 决定：M1 从零重写 `solvers/`，不在其上打补丁。
2. DiffusionDet 是 **x0 参数化**（`objective='pred_x0'`），且 VP 满足 `α²+σ²=1`
   → 与 `NoiseScheduleVP` + `model_type="x_start"` 天然对接，DPM-Solver-v3 移植阻力小。
3. `backbone features` 只在 `detector.py:313` 算一次 → **NFE 就是 head forward 次数**，成本模型简单。
4. Box Renewal（`detector.py:209-238`）+ Ensemble 仅在多步时生效（`detector.py:239`）
   → "多步更好"是混淆的，必须按 §2.3 FEP 解耦。
5. PFGM++ 相对 EDM 只改了**两处**（扰动核 + 先验），网络/预条件完全一致 → 性价比最高的一条线。

**下一步（等待指令即进入 M0）**
1. 建 `configs/lab/sdd.res50.yaml` + `diffusiondet/data_register.py`（修正背景类 id 映射）
2. 修 `train.py` 硬编码路径 → CLI 参数
3. 跑 `--eval-only` 冒烟 + 100 iter 训练，采集 `it/s`、`latency_ms`、`peak_mem_GB`

---

## 7. 阻塞与待决问题

| # | 问题 | 状态 | 影响 |
|---|---|---|---|
| Q1 | 是否同意"阶段 A 用 SDD 筛选、阶段 B 用 PLS 确认"的两阶段协议？ | 待用户确认 | M4/M5 排期 |
| Q2 | 是否在 M0 就先冻结/隔离 `detector_dpm3.py`、`detector_noise.py`、`dmp3.py`、`train.py`？ | 待用户确认（建议隔离） | M0 |
| Q3 | 是否允许修改原 `configs/*.yaml`？（建议一律新建到 `configs/lab/`） | 待用户确认 | M0 |
| Q4 | `petrain/diffdet_coco_res50.pth`（COCO 80 类）是否用作迁移初始化？ | 待用户确认（默认不用） | M4/M5 训练时长 |
| Q5 | 评测 seed 数量固定为 3 是否可接受？（影响 eval 总时长约 3×） | 待用户确认 | M2/M3 排期 |

---

## 8. 结论落袋区（随 H1–H4 判定推进，M6 汇总）

> 暂空。每个 hypothesis 判定完成后在此写一句话结论 + 指向 `RESULTS.md` 的数据表编号。
