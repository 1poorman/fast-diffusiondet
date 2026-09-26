# V2 蓝图：扩散机制本体优化（超越求解器替换）

> 版本 v1.0 ／ 2026-09-26
> 前置：v1 系列（`docs/BLUEPRINT.md`，H1–H4 已终判）证明"只换求解器"天花板极低
> （DPP@2 vs DDIM@1：+0.55 → +0.16 → +0.09 随训练增强递减）。
> 本蓝图转向**扩散机制本身**：训练-推理轨迹对齐、SNR 重设计、蒸馏、自适应早退。
>
> 范围：**仅 SDD**（筛选集，天花板 ~70 AP 有区分度）；bs 可加大以充分利用显存。
> 实时状态：`docs/v2/PROGRESS.md`；数据：`docs/v2/RESULTS.md` + `results/raw/v2_*.csv`

---

## 0. TL;DR

1. 机理假设：box 域 PF-ODE **轨迹近似直线**（x0 参数化 + 强图像条件），求解器离散误差
   非主导 → 解释 H1 收益微弱。**V0 用轨迹直线性诊断定量确证**。
2. 优化主轴从"沿轨迹怎么走"改为"把单步质量推到极限"：
   - **V1 训练-推理 t 对齐**（mix_tmax：50% 概率直接训 t=999，即 NFE=1 唯一访问点）
   - **V2 self-conditioning**（论文留了开关没启用；少步采样质量的公认增强器）
   - **V3 SNR_SCALE 重扫**（论文 COCO 先验 scale=2，从未在本数据验证）
   - **V4 t_mid 网格搜索**（training-free，只优化 DPP@2 甜点）
   - **V5 一致性蒸馏**（DPP@2 教师 → NFE=1 学生，扩散加速的正统路线）
   - **V6 自适应早退**（anytime 属性变现：高置信图 1 步、低置信图 2 步）
3. 已证伪不再碰：EMS 真统计量（H3）、EDM 训练范式（H4）、PFGM++（依赖 EDM 底座）。

---

## 1. 机理分析（为什么求解器路线到顶了）

### 1.1 原论文机制拆解

| 论文选择 | 实现 | 对优化的含义 |
|---|---|---|
| box 域低维扩散 `R^{B×P×4}` + SNR_SCALE=2 | `detector.py` q_sample | 轨迹维度低，弯曲有限 |
| pred_x0 参数化（head 直接回归绝对框） | `model_predictions` | 每步输出即检测结果（anytime）；单步直出被损失直接优化 |
| 训练随机均匀采 t | `prepare_diffusion_concat`（`torch.randint`） | **与推理访问的 t 网格错位**——V1 的切入口 |
| 推理 DDIM(eta=1)+renewal+ensemble | `ddim_sample` | 随机机制，与 ODE 语义冲突（v1 已解耦） |

论文自身数据已预示多步收益弱：COCO Res50 45.5@1→46.1@4（+0.6），Res101 仅 +0.3。
模型越强训练越足，多步越无益——与本仓库三数据集观察一致。

### 1.2 误差分解假设

```
总误差 = 回归误差（head 预测能力，训练侧决定） + ODE 离散误差（求解器侧决定）
```
NFE=1 已达可达 AP 上界附近 → 回归误差主导；求解器能压的只有离散误差项，
该项在低维直线轨迹上可忽略。**V0 诊断 = 直接测量离散误差占比。**

---

## 2. 假设与 Gate

| ID | 假设 | 判定 Gate | 状态 |
|---|---|---|---|
| **H5** | box 域 ODE 轨迹近似直线：DDIM(eta=0)@k=10 轨迹上各步 x0 预测与终点偏差小 | 噪端 t≥700 的相对偏差 <10% → H5 支持（求解器方向永久关闭）；>20% → 保留 V4 | ⬜ V0 |
| **H6** | 训练 t 分布对齐推理网格（mix_tmax p=0.5）使 NFE=1 AP 提升 ≥ +0.3 | 对同预算 uniform 控制臂，3 seed ΔAP ≥ +0.3 | ⬜ V1 |
| **H7** | self-conditioning 提升 NFE≤2 | ΔAP ≥ +0.3 | ⬜ V2 |
| **H8** | SNR_SCALE≠2（{1,4}）在 SDD 上改变单步-多步权衡 | 任一值 NFE=1 AP 超对照 +0.3 | ⬜ V3 |
| **H9** | DPP@2→1 步蒸馏使单步 AP ≥ DPP@2 − 0.3 | ≥ 70.0（对 bs16 对照重估） | ⬜ V5 |
| **H10** | 自适应早退：平均 NFE ≤1.3 时 AP ≥ DPP@2 − 0.3 | 同上 | ⬜ V6 |

基线参照（bs8 long，AP 已入库 `results/raw/m8_sdd_long.csv`）：ddim@1=70.27，DPP@2=70.36。

---

## 3. 验证步骤与排期（bs16 加速，SDD only）

| 步骤 | 内容 | 训练 | 评测 | Gate |
|---|---|---|---|---|
| **V0** | 轨迹直线性诊断（`scripts/diag_trajectory.py`） | 无 | ~5 min | H5 |
| **V1** | t 对齐：uniform 控制 @bs16 vs mix_tmax 处理 @bs16，双卡并行 | ~35 min×2 | ~10 min×2 | H6 |
| **V2** | self-conditioning（`self_condition=True`，head 输入加 x0 上一步预测） | ~40 min | ~10 min | H7 |
| **V3** | SNR_SCALE ∈ {1,4}（叠加 V1 赢家） | ~35 min×2 | ~10 min×2 | H8 |
| **V4** | DPP@2 的 t_mid 网格搜索（training-free，复用 patch 机制） | 无 | ~1 h | 可选 |
| **V5** | 蒸馏（教师=DPP@2 输出，学生=NFE=1 微调） | ~40 min | ~10 min | H9 |
| **V6** | 早退级联原型 + 吞吐实测 | 无 | ~1 h | H10 |

### 训练预算（bs16）

与 m8-long 相同 img-passes（1750 img × 18 epoch = 32000）：
- bs 8→16（线性规则 LR 1e-4→2e-4），4000→**2000 iter**，STEPS 1600（80%）
- 实测 bs8 ≈0.55 s/it → bs16 预计 ~1.0 s/it → **2000 iter ≈ 33 min/arm**
- 双卡并行：V1 两个 arm 一次跑完；对照臂（uniform@bs16）必须先于/同步于处理臂（FEP #7）

### 评测协议（沿用 FEP）

- 3 seed（0/1/2）、`BOX_RENEWAL=False + USE_ENSEMBLE=False`、同 NFE 同 seed 噪声
- 评测配置：`solvers ddim,dpm_v3` × `nfes 1,2` × 3 seed（8 evals/arm，~10 min）
- eval batch=32（显存富余，评测加速）
- 数字只记 `docs/v2/RESULTS.md`，CSV 落 `results/raw/v2_*.csv`

---

## 4. 改造点地图

| 目标 | 位置 | 改动 |
|---|---|---|
| 轨迹诊断脚本 | `scripts/diag_trajectory.py` | 新增（V0） |
| 训练 t 分布 | `detector.py:sample_train_t` + `config.py` | `TRAIN_T_DIST`/`TRAIN_T_HIGH_PROB`（V1） |
| V1 配置 | `configs/lab/v2/sdd.res50.bs16.{uniform,tmix}.yaml` | 新增 |
| self-conditioning | `head.py`（输入 concat）+ `detector.py` | V2，`self_condition=True` 路径 |
| t_mid 网格 | `scripts/run_m2_matrix.py` 扩展或新脚本 | V4 |
| 蒸馏 | `train_net.py` + 新 loss（教师框做 soft target） | V5 |
| 早退 | `solvers/` 新增 `adaptive.py` | V6 |

---

## 5. 风险

| ID | 风险 | 缓解 |
|---|---|---|
| R1 | mix_tmax 让模型偏科 t=999，DPP@2（需 t~500）退化 | 评测同时看 NFE=1 与 NFE=2；若 NFE=2 掉 >0.5，降 p 至 0.3 |
| R2 | bs16+LR2e-4 短周期调度敏感（v1 R15） | 控制臂同配置对照，处理-控制差值才有意义；异常退回 bs8 |
| R3 | 共享机器：GPU2/3 被 vLLM 占满 | 只用 GPU0/1；启动前 nvidia-smi 快照 |
| R4 | self-conditioning 改 head 输入 shape，迁移权重加载失败 | 新输入投影层 strict=False 初始化，不影响 trunk 加载 |
