# DiffusionDet 高效扩散机制对比实验 —— 技术蓝图

> 目标：把 `fast-diffusiondet` 的 DDIM 推理替换为更高效的采样机制，系统性地对比
> **DDIM（基线） / EDM（Karras + Heun） / PFGM++（重尾泊松流扰动） / DPM-Solver-v3（EMS 高阶多步）**
> 四种机制在**同等 NFE 预算**下的检测精度与推理耗时，并给出可复现的实验结论。
>
> 文档版本：v1.0 ／ 创建日期：2026-09-23 ／ 状态：待评审
> 实时进度见同目录 `PROGRESS.md`，实验数据见 `RESULTS.md`（M2 起开始填充）

---

## 0. TL;DR（先给结论，细节在后）

1. **可行，且三条机制可以正交组合。** DiffusionDet 的 box diffusion 在数据形式上是一个
   `x ∈ R^{B×P×4}`（`P=num_proposals`）的低维连续扩散问题，本质与图像扩散同源，
   DPM-Solver-v3 / EDM / PFGM++ 的核心 ODE 推导**都不依赖 image 结构**，可以直接迁移。
2. **分工是清晰的：**
   - **DPM-Solver-v3 / EDM Heun = 推理侧求解器**（training-free，无需重训即可出结果，收益最快）
   - **EDM = 训练侧参数化 + 噪声 + 加权范式**（需重训，改善训练动力学与上界）
   - **PFGM++ = 训练侧扰动核**（需重训，通过重尾扰动把 ODE 轨迹"拉直"，**直击少步采样**）
   三者可叠加：PFGM++ 决定"轨迹形状"，EDM 决定"参数化/加权"，求解器决定"如何沿轨迹走"。
3. **最大技术风险不是数学，而是 DiffusionDet 特有的 Box Renewal 机制**——它在每步裁掉低分框并随机补齐，
   会**破坏多步求解器依赖的历史缓存与 ODE 轨迹语义**。必须在实验协议里显式处理（见 §7 R1）。
4. **硬件是关键约束：单卡 RTX 3060 Laptop 6GB。** 必须用"两阶段实验协议"（小集筛选 → 大集确认），
   否则单条训练曲线就要十几个小时，全矩阵不可行。见 §2.4 与 §5 M4/M5。
5. **当前仓库里有一份半成品的 DPM-Solver-v3 移植（`detector_dpm3.py`），存在必崩的 AttributeError，
   不能直接跑。** 详见 §1.4 的需要先清理的坑。

---

## 1. 现状勘查（已实测，非推测）

### 1.1 运行环境（实测输出）

| 项 | 实测值 | 备注 |
|---|---|---|
| OS | win32 (Windows) | DiffusionDet 官方仅声明 Linux/macOS，Windows 需自行保证 detectron2 可用 |
| conda env | `detectron2` → `D:\APP\conda\envs\detectron2` | 已存在，直接用 |
| Python | **3.8.17** | |
| PyTorch | **1.10.0 + cu113**，`torch.cuda.is_available()=True` | ⚠ 低于 EDM/PFGM++ 官方仓库要求的 1.12 |
| detectron2 | **0.6**，源码装载自 `e:\files\dl-code\models\detectron2\detectron2` | 与 `train_net.py` 的 `AMPTrainer` API 匹配，版本正确 |
| numpy | 1.24.4 | |
| GPU | **NVIDIA GeForce RTX 3060 Laptop**，**6.0 GB** | 单卡，显存是硬瓶颈 |

> ⚠ **torch 1.10 的直接影响**：`torch.autograd.forward_ad`（前向模式自动微分）在 1.11 才引入。
> DPM-Solver-v3 官方的 EMS 统计量计算脚本 `compute_EMS_scoresde.py:128-156` 依赖它。
> → **本项目必须改用有限差分 JVP 实现 EMS**（见 §3.4）。已在 M3 中列为必办事项。

### 1.2 可用数据（实测）

| 数据集 | 绝对路径 | train | val | thing 类数 | 备注 |
|---|---|---|---|---|---|
| **PLS / Plantv2** | `e:/Files/DL-code/model+/diffusion-model/dataset/PLS/Plantv2` | 7916 img / 7915 ann | 2024 img / 2024 ann | **16** | 另有 test2017 111 张且**带标注**（`instances_test2017.json`）。平均 ~1 实例/图 |
| **SDD / Strawberry** | `e:/Files/DL-code/model+/diffusion-model/dataset/SDD/Strawberry` | 1750 img / 3932 ann | 750 img / 1775 ann | **7** | 实例较密，适合做找回率差异的观测 |

数据格式均为 **COCO json**，目录结构 `train2017/ val2017/ annotations/instances_{train,val}2017.json`，detectron2 可直接吃。

> ⚠ **类别 id 陷阱（已实测）**：两个 json 的 `categories` 都含 `(0, '_background_')`，
> 但 `annotations` 的 `category_id` **从不等于 0**。
> → 必须**显式构造** `thing_dataset_id_to_contiguous_id = {cid: cid-1 for cid in 1..K}`，
> 设置 `thing_classes` 为 K 个真实类名，**`NUM_CLASSES = K`**（PLS=16, SDD=7）。
> 不能依赖 detectron2 默认的 `enumerate(cat_ids)` 映射，否则会多出一个永远无 GT 的空类，
> 污染 Focal loss 的先验与 mAP 分母。

### 1.3 fast-diffusiondet 关键机制定位

**包路径**：`e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet/diffusiondet/`

| 机制 | 文件:行 | 现状 |
|---|---|---|
| 噪声调度 `cosine_beta_schedule` | `detector.py:49-59` | T=1000，`s=0.008` |
| 扩散超参初始化 | `detector.py:83-128` | `objective='pred_x0'`、`ddim_sampling_eta=1.`、`self_condition=False`、`scale=SNR_SCALE=2.0` |
| GT 加噪 `prepare_diffusion_concat` | `detector.py:370-405` | 实际使用的那个（在 `detector.py:419` 被调用）；GT + `randn/6+0.5` 占位补齐至 `num_proposals` |
| 前向加噪 `q_sample` | `detector.py:277-284` | 标准 VP：`x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` |
| **模型输出转换 `model_predictions`** | `detector.py:170-184` | **x0 参数化**：head 输出绝对坐标 → 归一化 → `(x*2-1)*scale`；再由 `predict_noise_from_start`（`detector.py:164-168`）反算 ε |
| **DDIM 采样主循环 `ddim_sample`** | `detector.py:186-274` | 见下方三段关键代码 |
| DDIM 核心更新 | `detector.py:220-234` | `img = x_start*√α_next + c*pred_noise + σ*noise` |
| Box renewal | `detector.py:209-219, 236-238` | 分数>0.5 裁剪 + `randn` 补齐（**多步求解器的最大障碍**） |
| Ensemble + NMS | `detector.py:239-274` | **仅当 `sampling_timesteps > 1` 时启用**（`detector.py:239, 247`） |
| 推理入口 | `detector.py:313-315` | **`backbone features` 只算一次**，之后每步只用 head → NFE 只计 head forward |
| 时间步嵌入 | `head.py:31-43, 96-101, 271-274` | `SinusoidalPositionEmbeddings(256)` → MLP(1024) → 各 stage FiLM 调制 |
| 配置项 | `config.py:11-77` | `SNR_SCALE=2.0`、`SAMPLE_STEP=1`、`NUM_PROPOSALS=300`、`USE_NMS=True` |
| 损失 | `loss.py` | `HungarianMatcherDynamicK` + `SetCriterionDynamicK`：Focal + L1 + GIoU，**不是 ε-MSE** |

```220:234:e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet/diffusiondet/detector.py
            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)    ## sample noise

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise
```

**关于"步数长耗时久"的精确定位**：默认 `SAMPLE_STEP=1`（`config.py:52`），此时 `times=[999,-1]`、
只有一次 head forward，`use_ensemble` 因 `sampling_timesteps>1` 不成立而关闭。
想提精度就要 `SAMPLE_STEP=4/8`，此时 head forward 次数线性增长，且**额外叠加了 ensemble+NMS 的收益**，
导致"多步为什么更好"这件事本身是混淆的。**这必须在实验协议中解耦**（见 §2.3）。

### 1.4 需要先清理的坑（当前 repos 自带）

| # | 严重度 | 位置 | 问题 | 处理 |
|---|---|---|---|---|
| B1 | 🔴 | `detector_dpm3.py:624, 641` | 调 `self.multistep_{predictor,corrector}_update`，但 `Dpm3Det` 未持有 `DPM_Solver_v3` 实例，这两个方法只存在于 `samplers/dpm_solver_v3.py:445, 488` | **不要在它上面改**，M1 重写；仅作为移植参考 |
| B2 | 🔴 | `detector_dpm3.py:738` | `forward()` 调 `self.ddim_sample(...)`，但该类没有 `ddim_sample` | 同上 |
| B3 | 🔴 | `detector_dpm3.py:426-467` | `self.noise_schedule` 从未赋值（`__init__` 参数是字符串），`total_N` 访问必崩 | 同上 |
| B4 | 🟡 | `detector_dpm3.py:433-435` | 依赖外部 `l.npz`/`sb.npz`，仓库内**不存在** | M3 自己算 |
| B5 | 🟡 | `diffusiondet/samplers/` | **缺 `__init__.py`**，包内相对导入不可用 | M1 建新包 `solvers/` 时补上 |
| B6 | 🟡 | `detector.py:62`、`detector_noise.py:90`、`dmp3.py:21` | 三处同名注册 `"DiffusionDet"`，registry 被最后 import 者覆盖 | `__init__.py` 未导出后两者，暂时无害；M0 清理 `dmp3.py`（残缺 stub，第 30 行后无 forward、且 `cfg.MODEL.DIFFUSIONDET` 大小写错误） |
| B7 | 🟠 | `train.py:42` | 硬编码 `E:\Files\DL code\datasets\wgisd`（非 raw string，路径也不存在） | M0 改为 CLI 可配 + 注册 PLS/SDD |
| B8 | 🟠 | `configs/diffdet.coco.res50.yaml:8-12` | 文件名叫 coco，实际是 WGISD 8 类 | M0 另建 `configs/lab/*.yaml`，不动原文件 |
| B9 | 🟡 | `configs/Base-DPM3Det.yaml` | 无任何 yaml 引用它，孤立 | M1 起接管或删除 |

---

## 2. 问题定义、指标与实验协议

### 2.1 统一符号

| 符号 | 含义 |
|---|---|
| `x` | 归一化后的 box 张量，形状 `(B, P, 4)`，值域 `[-scale, scale]`，`scale=SNR_SCALE=2` |
| `P` | `NUM_PROPOSALS`（默认 300，res50 配置 500） |
| `D(x; t)` | head 给出的 **x0 预测**（即 `model_predictions` 的 `x_start`），不是 ε |
| `NFE` | **head 的前向调用次数/图**（backbone 只算一次，不计入） |
| `α_t, σ_t` | VP 的 `√ᾱ_t, √(1-ᾱ_t)`，满足 `α²+σ²=1` |

### 2.2 度量指标（全部必须同时采集）

| 类别 | 指标 | 定义 / 采集方式 |
|---|---|---|
| **质量（主）** | `AP`, `AP50`, `AP75`, `AP_S/M/L` | detectron2 `COCOEvaluator`，val split |
| **质量（辅）** | `AR@100` | 找回率，对增资采样机制敏感 |
| **成本（主）** | `NFE` | 帧内 head forward 次数，代码内计数器取数而非推算 |
| **成本（主）** | `latency_ms` | 单图 batch=1，warmup 20 张、计时 200 张取**中位数**，`torch.cuda.synchronize()` 前后包围 |
| **成本（辅）** | `peak_mem_GB` | `torch.cuda.max_memory_allocated()` |
| **训练成本** | `train_wallclock_h`, `it/s` | M0 先测出 `it/s` 基线，用于排期 |

### 2.3 公平对比协议（FEP —— Fair Evaluation Protocol，极其重要）

不遵守这条，所有数字都不可信：

1. **同 checkpoint 对比**：training-free 系列（M2/M3）必须加载**同一个** baseline checkpoint，权重完全冻结。
2. **解耦 ensemble**：`detector.py:239` 的 `use_ensemble and sampling_timesteps > 1` 会把多步天然加成。
   → 主矩阵在 **`BOX_RENEWAL=False` + `ENSEMBLE=False`** 下跑，得到**纯采样器的 AP-NFE 曲线**；
   ensemble 与 box renewal 单独作为一组 ablation（A1/A2）给出，明确标注"+ENS"/"+BR"。
3. **固定随机初值**：同一 NFE 下不同 solver 使用**同一个 `torch.randn` 种子**（可注入 `noise` 参数），
   消除 Monte-Carlo 噪声；每个配置跑 3 seed 取均值±std。
4. **同 NMS / 同 proposal 数**：`USE_NMS=True`、阈值 0.5、`NUM_PROPOSALS` 全矩阵一致。
5. **训练侧等预算**：所有需要重训的 arm 使用**相同的 MAX_ITER / batch / LR schedule / seed / 数据增广**。
6. **数字只记一处**：所有结果实时写入 `RESULTS.md` 的表格，脚本自动 append CSV（`results/raw/*.csv`），
   禁止手工抄写到多处。

### 2.4 两阶段实验协议（因 6GB 单卡而设计）

- **阶段 A（筛选）**：在 **SDD/Strawberry** 上做短周期训练（`MAX_ITER` 目标 ~4 epoch ≈ 3500 iter @ bs=2），
  跑**全 matrix**。用于剔除无效分支。
- **阶段 B（确认）**：只让阶段 A 的 **top-2 组合 + baseline** 在 **PLS/Plantv2** 上跑完整训练，产出论文级数字。

> 节奏预算在 M0 实测 `it/s` 之后定稿，写进 `PROGRESS.md` 的"排期"表。

---

## 3. 技术方案

### 3.1 统一抽象层（M1 产出）

新建包 `e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet/diffusiondet/solvers/`
（**必须带 `__init__.py`**）：

```
solvers/
├── __init__.py              # SAMPLER_REGISTRY + build_solver(cfg)
├── base.py                  # DenoiseFn 协议 + BoxSolver 基类 + NFE 计数器
├── schedule.py              # VPVPSchedule / EDMSchedule：α(t),σ(t),λ(t),λ⁻¹,round_sigma
├── ddim.py                  # 从 detector.py 原样搬出，行为必须逐 bit 一致
├── heun.py                  # EDM-eq Heun（含 Karras ρ=7 时间步）
├── dpm_solver_v3.py         # 移植自 DPM-Solver-v3-main/dpm_solver_v3.py
├── statistics.py            # EMS 统计量计算（有限差分 JVP 版）
└── renew.py                 # Box renewal 与 history 失效策略（M6）
```

**唯一的去噪函数接口**（把 detector 的 knowledge pair 包成纯函数）：

```python
def denoise_fn(x, sigma_or_t) -> x0_hat      # (B,P,4) -> (B,P,4)
```

它内部完成 `detector.py:170-184` 的三件事：clamp→反归一化→xyxy→乘 whwh→`head(...)`→反变换回来。
**除 DDIM 外所有 solver 都必须"看不见" `images_whwh` / `backbone_feats` / `batched_inputs`**，
这样三个 solver 可以互相对拍。

### 3.2 基线 S0：DDIM（保持不动）

直接把 `detector.py:186-274` 抽成 `solvers/ddim.py`。
**验收硬指标（M1 Gate）**：重构前后在同一 seed 下输出的 box 坐标最大绝对误差 `< 1e-5`。

### 3.3 机制 A：EDM（来自 `edm-main`）

EDM 给我们**两件东西**，必须拆开单独评估：

**(A-1) 采样器**（training-free，直接从原仓库搬）—— `edm-main/generate.py:25-60`：

```25:60:e:/Files/DL-code/model+/diffusion-model/edm-main/generate.py
def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    ...
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) *
               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    ...
        denoised = net(x_hat, t_hat, class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur
        if i < num_steps - 1:                              # Heun 二阶校正
            denoised = net(x_next, t_next, class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
```

要点：`NFE = 2N-1`；`S_churn=0` 时完全确定性；`σ` 网格是 **Karras ρ=7 幂律**（`generate.py:36`）。

> **移植到 VP 模型的低风险路径**：不要自己推 VP→σ 空间公式，直接搬 EDM 的**通用消融采样器**
> `edm-main/generate.py:66-176`（`ablation_sampler`），它已内置
> `solver∈{euler,heun}`、`discretization∈{vp,ve,iddpm,edm}`、`schedule`、`scaling` 四个正交开关。
> 钉 `discretization='vp', schedule='vp', scaling='vp'` 即可在不重训的前提下给现有 checkpoint 跑 Heun，
> 再钉 `solver='euler'` 得到同 NFE 参照组。**这是 M2 的主力实现路径，数学风险最小。**

**(A-2) 训练范式**（需重训）—— `edm-main/training/loss.py:65-80` + `training/networks.py:660-663`：

```
训练: σ ~ LogNormal(P_mean=-1.2, P_std=1.2);  x = x0 + σ·ε
预条件: c_skip = σ_data²/(σ²+σ_data²)
        c_out  = σ·σ_data/√(σ²+σ_data²)
        c_in   = 1/√(σ_data²+σ²)
        c_noise= ln(σ)/4
前向:  D(x;σ) = c_skip·x + c_out·F(c_in·x, c_noise)
损失加权: λ(σ) = (σ²+σ_data²)/(σ·σ_data)²
```

**落到 DiffusionDet 的映射困难点（必须设计后验证）**：
- DiffusionDet 的 loss 不是 MSE，而是匈牙利匹配后的 `Focal + L1 + GIoU`（`loss.py`）。
  → EDM 的 `λ(σ)` 不能整体系数化套上去，正确做法是**只对 regression 分支（L1+GIoU）乘 λ(σ)**，
  classification（Focal）保持原样。这条要在 M4 里写成 ablation A4。
- `σ_data` 必须**实测**：统计训练集 `prepare_diffusion_concat` 输出 `x_start` 的逐坐标标准差作为初值
  （不是 EDM 图像默认 0.5）。写一个一次性脚本 `scripts/calib_sigma_data.py` 产出 `sigma_data.json`。
- `c_in·x` 意味着喂给 head 的**不再是 clamped 后的归一化框**，尺度变了 → 与 `SNR_SCALE` 的作用重复，
  二者**只能取一**：建议 M4 期间把 `SNR_SCALE` 语义改为"数据尺度"，在 `c_in/c_out` 中显式体现，并做 A5 ablation。
- 时间嵌入：`head.py:96-101` 的 `SinusoidalPositionEmbeddings` 输入原本是 `0..999` 的整数，
  改成 `c_noise=ln σ/4 ∈ [-1.55, 1.10]` 后动态范围剧变 → 必须做 A6 ablation：`Sinusoidal` vs
  `GaussianFourierProjection`（`head.py:46-57` 已有实现）以及是否加 `4σ` 缩放因子。

### 3.4 机制 B：DPM-Solver-v3（来自 `DPM-Solver-v3-main`）

**好消息**：DiffusionDet 的 VP 前向（`detector.py:277-284`）与 `NoiseScheduleVP` 的假设**完全吻合**。
`DPM-Solver-v3-main/dpm_solver_v3.py:196-200` 硬编码 `σ_t = √(1-α_t²)`，而我们正好是 `α²+σ²=1`；
且 `objective='pred_x0'`（`detector.py:86`）对应 `model_wrapper(model_type="x_start")`
（`dpm_solver_v3.py:239-242, 361`）。所以：

```python
ns = NoiseScheduleVP(schedule="discrete", betas=cosine_beta_schedule(1000))   # 与 detector.py:87 同源
model_fn = model_wrapper(head_denoise_fn, ns, model_type="x_start")           # 我们就是 x0 预测
dpm = DPM_Solver_v3(statistics_dir=..., noise_schedule=ns, steps=k,
                    skip_type="logSNR", degenerated=False, device="cuda")
x0 = dpm.sample(x_T, model_fn, order=3, p_pseudo=False, use_corrector=True,
                c_pseudo=True, lower_order_final=True)
```

**必须与已知坑对抗的处理**：

| 坑 | 来源 | 对策 |
|---|---|---|
| `model_wrapper` 总是三 positional 调用 `model(x, t, cond)` | `dpm_solver_v3.py:356, 358` | 包装器第三参接 `None` 并忽略 |
| Statistics 形状 `(N+1, C, H, W)` 与数据强绑定 | `compute_EMS_scoresde.py:194` 的 `mean(dim=0)` | **改用 our own**：见下 |
| `.cuda()` 硬编码 | `dpm_solver_v3.py:500, 511-518, 661, 706` | 我们有 GPU，可接受；但 Keep `device` 一致性，不做 CPU CI |
| `NoiseScheduleEDM` 无 `total_N/T` | `dpm_solver_v3.py:8-38` | 我们走 VP 路径，不需要 |
| EMS 依赖 `forward_ad` | `compute_EMS_scoresde.py:11` | **torch 1.10 没有** → 有限差分 JVP |
| `order/p_pseudo/use_corrector/c_pseudo/lower_order_final` 无默认值 | `dpm_solver_v3.py:740-750` | 全部显式传，进 `cfg` |

**EMS 统计量的本项目适配（M3 核心）**：
原版对 batch 维取均值得到逐像素统计量。本项目 `x ∈ (B,P,4)`，
且 proposal 之间**可交换**（推理时全部由 `randn` 初始化，无位置语义），因此允许降维：

| 模式 | 统计量形状 | 说明 | 与 `(B,P,4)` 的广播 |
|---|---|---|---|
| `scalar` | `(N+1, 1, 1)` | 所有坐标共用 | ✅ 直接广播 |
| `per_coord` | `(N+1, 1, 4)` | cx/cy/w/h 各自统计（**推荐**，w/h 与 cx/cy 先验明显不同） | ✅ 直接广播 |
| `per_prop` | `(N+1, P, 4)` | 原版等价，**不推荐**（与 renewal 冲突、且内存 `(1001,500,4)` 尚可但无收益） | ✅ 但不稳定 |

有限差分 JVP 实现（`statistics.py`）：对每个 logSNR 网格点 `i`、随机方向 `v ~ N(0,I)`（形状同 `x`）：

```
eps_fd ≈ (D(x+h·v; σ_i) - D(x-h·v; σ_i)) / (2h),   h ≈ 1e-3·scale
l_i = mean_over_batch_and_coords( σ_i · eps_fd · v )
```

再按 `dpm_solver_v3.py` 的方式积分出 `s, b`（沿用原脚本的 `weighted_cumsumexp_trapezoid`，
路径 `dpm_solver_v3.py:413-437`，可直接 `import`）。最终产出 `statistics/{tag}/{sigma}_{steps}_{K}/l.npz` + `sb.npz`。

**`degenerated=True` 是关键对照组**（`dpm_solver_v3.py:492-495` 把 `l=1,s=0,b=0`）：
它等价于 **DPM-Solver++**，**不需要真实 EMS 统计量**，可以在 M2 就出结果，把 M3 的收益单独量化出来。

### 3.5 机制 C：PFGM++（来自 `pfgmpp-main`）

**重大发现（已核实）**：本仓库的 PFGM++ 是 **NVlabs EDM 的 fork**，其 sampler 里**没有任何 z 状态、
没有散度估计、没有 Hutchinson / BFSD**（全仓正则检索 `divergence|get_div|BFSD|kappa` 命中数为 0）。
它相对 EDM **只有两处不同**：

1. **训练扰动核**（`pfgmpp-main/training/loss.py:111-145`）：
   ```120:138:e:/Files/DL-code/model+/diffusion-model/pfgmpp-main/training/loss.py
   r = sigma.double() * np.sqrt(self.D).astype(np.float64)     # r = σ√D
   samples_norm = np.random.beta(a=self.N / 2., b=self.D / 2., size=images.shape[0])
   samples_norm = np.clip(samples_norm, 1e-3, 1-1e-3)
   inverse_beta = samples_norm / (1 - samples_norm + 1e-8)
   samples_norm = r * torch.sqrt(inverse_beta + 1e-8)
   gaussian = torch.randn(images.shape[0], self.N)
   unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
   perturbation_x = unit_gaussian * samples_norm     # 重尾径向 × 均匀方向
   ```
   即：把各向同性高斯 `σ·ε`（半径分布 `χ_N`）换成 **Beta′(N/2, D/2) 重尾径向分布**。
2. **采样先验**（`pfgmpp-main/generate.py:203-227`）同样用 Beta′ 采样，`pfgmpp=True` 时**不再乘 σ_max**
   （`generate.py:47-50`），因为 `latents` 已含 `r = σ_max·√D` 的尺度。

**网络结构、preconditioning、σ 分布全部与 EDM 一字不差**（`networks.py:700-703` vs EDM `660-663`）。
这意味着：**PFGM++ = EDM + 两行扰动核改动**。对我们而言这是**高性价比**的第三条线。

**落到 box 空间的映射**：
- 数据维数 `N = 4 × P`（P=300 → N=1200）。注意 Beta′(N/2, D/2) 在 `N ≥ 1200` 时会**极度集中在均值附近**
  → `D` 必须足够大才有效。EDM 论文里 CIFAR-10 (N=3072) 的 sweet spot 是 `D=128/2048`。
  → 建议扫描 **D ∈ {32, 128, 512, 2048, ∞(即纯 EDM)}**，在阶段 A 上选。
- ⚠ **Beta 采样是 NumPy 的**（`loss.py:123`），每 step 一次 `np.random.beta` —— 需要确认它在
  Windows 上不会成为 dataloader worker 的瓶颈；建议预先生成一个大表（如 1e6 个）循环取用。
- ⚠ **radial 归一化是对整个 `(B, N)` 展平后做的**（`loss.py:134-137`），等价于把**一张图的所有 proposal 的 4 个坐标**
  当成一个 N 维向量整体归一化。这与 DiffusionDet 里"proposal 之间相互独立"的假设不同！
  → 设计两种粒度做 ablation A7：
  - `global`：整幅图所有 proposal 一起做 radial 归一化（忠于原论文）
  - `per_proposal`：每个 proposal 的 4 维单独 Beta′ 采样（更贴合 DiffusionDet 的 proposal 独立性假设）
- `pfgmpp-target` 那条泊松场经验目标（`loss.py:229-254`）默认 **`stf=False` 时不启用**，
  目标就是干净 `x0`（`loss.py:167`），**不要碰它**。

### 3.6 三者的组合关系

```
训练侧 ──► 扰动核:    高斯(EDM/DDIM)   or   Beta′ 重尾(PFGM++, D=超参)
      └─► 参数化:    VP-DDPM                      or   EDM c_skip/c_in/c_out/c_noise
      └─► 加权:      无 / EDM λ(σ)（作用于 reg 分支）
推理侧 ──► 求解器:    DDIM(1阶)  Heun(2阶)  DPMv3-degenerated(≈DPM-Solver++)  DPMv3-EMS(高阶+EMS)
```

**研究假设（待验证，写死在这里以便事后对照）**：
- H1：NFE ≤ 4 时，2 阶及以上求解器（Heun / DPMv3）AP 显著优于 DDIM（≥ +0.5 AP）。
- H2：PFGM++ 的直线化轨迹使 2 步即可接近 DDIM 8 步的 AP（≥ -0.2 AP 内）。
- H3：EMS 真统计量相对 `degenerated` 仅在 NFE ≤ 5 时有可见收益。
- H4：EDM 参数化主要提升**上界**（高 NFE 的 AP），而非低 NFE 效率。

---

## 4. 实验矩阵

编号规则：`E{训练范式}-{采样器}`，NFE 一律取值 `{1,2,3,4,6,8,10}`。

| ID | 训练范式 | 采样器 | 训练成本 | 说明 |
|---|---|---|---|---|
| **E0-DDIM** | F0 VP-DDPM（原） | DDIM | 1×（基准） | **基线曲线** |
| E0-HEUN | F0 | Heun (ablation_sampler, vp/vp/vp) | 0 | training-free |
| E0-EULER | F0 | Euler | 0 | 剥离"阶数"与"时间步"收益 |
| E0-DPP | F0 | DPMv3 `degenerated=True` | 0 | ≈ DPM-Solver++ |
| E0-DPV3 | F0 | DPMv3 `degenerated=False` + EMS | 0 | **M3 主力** |
| E1-HEUN | F1 EDM 参数化 | Heun + Karras ρ=7 | ~1× | M4 |
| E1-DPV3 | F1 | DPMv3 + EMS | ~1× | M4（注意：EDM 训练后 σ 空间与 VP noise schedule 不同） |
| E2-HEUN | F2 PFGM++ (D*) | Heun + Beta′ 先验 | ~1× | M5，**预期最优** |
| E2-DPV3 | F2 | DPMv3 + EMS(per_coord) | ~1× | M5/M6 组合王牌 |

**Ablation（编号 A*）**

| ID | 内容 |
|---|---|
| A1 | Box Renewal 开/关 × 多步求解器（含 history-invalidation 策略） |
| A2 | Ensemble+NMS 开/关 |
| A3 | `SNR_SCALE ∈ {1,2,4}` 与新参数化的交互 |
| A4 | EDM `λ(σ)` 只作用于 reg 分支 vs 全 loss |
| A5 | `SNR_SCALE` 与 `c_in/c_out` 二选一 |
| A6 | 时间嵌入：Sinusoidal vs GaussianFourier，及 `c_noise=lnσ/4` 的缩放因子 |
| A7 | PFGM++ radial 粒度：global vs per_proposal |
| A8 | EMS 统计量模式：scalar vs per_coord vs none(degenerated) |
| A9 | Karras `ρ ∈ {3,7,12}` 与 `skip_type ∈ {logSNR, time_uniform}` |

---

## 5. 里程碑与验收标准（Gate）

> 每个 Gate **必须全部满足**才能进入下一里程碑。Gate 检查表打勾在 `PROGRESS.md`。

### M0 — 环境与基线建立 （优先级：最高，必须先过）

**任务**
1. 新建 `configs/lab/pls.res50.yaml` 与 `configs/lab/sdd.res50.yaml`，
   继承 `configs/Base-DiffusionDet.yaml`，修正 `NUM_CLASSES`（16 / 7）、`NUM_PROPOSALS=300`、
   `IMS_PER_BATCH=2`、`SOLVER.AMP.ENABLED=True`、`OUTPUT_DIR`。
2. 新建 `diffusiondet/data_register.py`：显式构造 `thing_dataset_id_to_contiguous_id`（排除 id=0），
   注册 `pls_{train,val,test}` 与 `sdd_{train,val}`。
3. 修正 `train.py` 的硬编码路径改为 CLI 参数（B7）；删除/隔离 `dmp3.py`、`detector_noise.py`（B6）。
4. 实机测 `it/s`、`latency_ms`、`peak_mem` 基线。
5. **在 SDD 上跑通短周期训练（约 3500 iter）并 eval，产出 baseline checkpoint。**
6. 编写 `run_all.sh` / `run_all.ps1` 一键跑完整矩阵。

**Gate M0**
- [ ] `python train_net.py --config-file configs/lab/sdd.res50.yaml --eval-only` 能跑出非空 AP（> 0）
- [ ] 训练 100 iter 无 OOM，log 中出现正常下降的 loss
- [ ] baseline checkpoint 在 SDD val 上 **AP 有量 > 0**（短周期即可，不要求高）
- [ ] `it/s`、`latency_ms(NFE=1)`、`peak_mem_GB` 三项已写入 `RESULTS.md`
- [ ] `PROGRESS.md` 中已据此给出后续里程碑的**排期估算**

### M1 — 采样器统一接口与数值正确性

**任务**
1. 抽出 `solvers/` 包与 `denoise_fn` 协议；DDIM 行为逐 bit 保持。
2. 实现 `heun.py`（搬 `ablation_sampler` 的 vp/vp/vp 配置）。
3. 移植 `dpm_solver_v3.py` + `NoiseScheduleVP`（`discrete`, betas 取自 `detector.py:87`），
   `model_wrapper` 用 `model_type="x_start"`。
4. 新增配置项 `MODEL.DiffusionDet.SOLVER`、`ORDER`、`SKIP_TYPE`、`DEGENERATED`、`STATS_DIR`。

**Gate M1（正确性优先于性能）**
- [ ] **DDIM 等价回归**：新 `ddim.py` 与旧 `detector.py:186-274` 在同 seed 下 box 输出 `max|Δ| < 1e-5`
- [ ] **解析解回归**：把 `denoise_fn` 换成一个**已知真值的线性/高斯混合去噪器**（可闭式求 ODE 解），
      三套 solver 各自误差随 NFE 呈**预期收敛阶**（Euler O(h)、Heun O(h²)、DPMv3 ≥ O(h²)），实测阶数偏差 `< 0.3`
- [ ] **NFE 计数正确**：三个 solver 的实测 head forward 次数 = 理论值（DDIM k / Heun 2N-1 / DPMv3 steps）
- [ ] 三个 solver 都能在真实 checkpoint 上跑出**有限**的 box（无 NaN/Inf），且 shape 全程 `(B,P,4)`

### M2 — Training-free 对比矩阵（**最先产出的可用结论**）

**任务**：用 M0 的 baseline checkpoint，跑 E0-{DDIM,HEUN,EULER,DPP} × NFE{1,2,3,4,6,8,10} × 3 seed。

**Gate M2**
- [ ] 四张 AP–NFE 曲线、四张 latency–NFE 曲线全部入 `RESULTS.md`（含 ±std）
- [ ] **主验收（H1）**：存在某个 `NFE ≤ 4` 的配置，其 AP 比同 NFE 的 DDIM **高 ≥ 0.5 AP**，
      或达到 DDIM@NFE=10 的 AP（`-0.2` 容差）所需 **NFE 降低 ≥ 2×**
- [ ] 单图 `latency_ms` 在同 NFE 下与 DDIM 相差 `< 15%`（证明没引入隐藏开销）
- [ ] 完成 A1 (renewal)、A2 (ensemble) 两组消融并给出结论

### M3 — EMS 统计量标定与 DPM-Solver-v3 全功率

**任务**：`statistics.py` 用有限差分 JVP 计算 `l/s/b`（`scalar` 与 `per_coord` 两版），
产出 `statistics/<dataset>/<tag>/{l.npz,sb.npz}`，跑 E0-DPV3。

**Gate M3**
- [ ] EMS 计算脚本在 SDD 上 1000 个 logSNR 网格点、K=256 样本，**单卡 6GB 下 < 2 小时**完成（否则降低 K/网格）
- [ ] `l/s/b` 数值无 NaN/Inf，`l` 随 t 单调光滑（相邻网格相对变化 `< 10`）
- [ ] **主验收（H3）**：`degenerated=False` 相对 `degenerated=True` 在至少一个 NFE ≤ 5 的点上 **AP 提升 ≥ 0.3**
- [ ] 完成 A8（三种统计量模式）消融

### M4 — EDM 训练范式移植

**任务**：引入 `c_skip/c_in/c_out/c_noise`、`σ~LogNormal(P_mean,P_std)`、`λ(σ)` 加权（仅 reg 分支）、
Karras ρ=7 时间步；`scripts/calib_sigma_data.py` 实测 `σ_data`。

**Gate M4**
- [ ] **等预算训练不退化**：相同 MAX_ITER 下，EDM 版在 NF=10 时的 AP **不低于** VP 基线的 `-0.5 AP`
- [ ] `σ_data`、`P_mean`、`P_std` 的标定过程与数值写入 `RESULTS.md`
- [ ] 完成 A4、A5、A6 三组消融
- [ ] **主验收（H4）**：EDM 版的高 NFE AP 上界相对基线提升 ≥ 0.5 AP，**或**给出的结论明确否定 H4（否定也是有效结论）

### M5 — PFGM++ 训练范式移植

**任务**：在 M4 的 EDM 基础上加 Beta′ 重尾扰动核 + 重尾先验；扫 `D ∈ {32,128,512,2048,∞}`。

**Gate M5**
- [ ] 训练 loss 稳定收敛（无 NaN，loss 曲线与 EDM 版同量级）
- [ ] **主验收（H2）**：存在某个 `D`，使得 **NFE=2 时的 AP ≥ EDM@NFE=8 的 AP − 0.5**
- [ ] 完成 A7 粒度消融
- [ ] 单个 `D` 的完整训练 wall-clock **不超过** EDM 版本的 1.1×

### M6 — 组合、确认与报告

**任务**：阶段 A（SDD）选出 top-2 组合，在 **PLS/Plantv2** 上跑完整训练核对；出版 `RESULTS.md` 终稿。

**Gate M6**
- [ ] PLS 上的结论与 SDD **方向一致**（否则需在报告中明确讨论跨数据集不稳定性）
- [ ] 所有 hypothesis H1–H4 均有明确 "支持/不支持/部分支持" 结论及数据支撑
- [ ] 最终推荐配置给出 **AP vs NFE vs latency** 三维 Pareto 前沿表
- [ ] `RESULTS.md` 中每张表都能由 `results/raw/*.csv` 复现

---

## 6. 改造点地图

| 目标 | 现有位置 | 计划改动 |
|---|---|---|
| 新增 solver 包 | — | 新建 `diffusiondet/solvers/`（含 `__init__.py`） |
| 去噪函数提取 | `detector.py:170-184` | 抽出闭包 `make_denoise_fn(self, backbone_feats, images_whwh)` |
| DDIM 迁移 | `detector.py:186-274` | 搬至 `solvers/ddim.py`，`detector.py` 保留薄包装调用 |
| Box renewal | `detector.py:209-219, 236-238` | 抽出 `solvers/renew.py`，加 cfg 开关；history 失效策略 |
| Ensemble | `detector.py:239-274` | 加 cfg 开关（现为硬编码 `True`） |
| 配置项 | `config.py:11-77` | 追加 `SOLVER/ORDER/SKIP_TYPE/DEGENERATED/STATS_DIR/BOX_RENEWAL/USE_ENSEMBLE/FORMULATION/PFGM_D/P_MEAN/P_STD/SIGMA_DATA/TIME_EMB` |
| 训练加噪 | `detector.py:370-405` | 按 `FORMULATION` 分支：VP / EDM / PFGM++ |
| 前向扩散 | `detector.py:277-284` | 增加 EDM σ 形式分支 |
| 损失加权 | `loss.py` | reg 分支乘 `λ(σ)`（EDM）/ 不变（PFGM++ 已在扰动里） |
| 时间嵌入 | `head.py:96-101` | 接入 `c_noise`，可选 `GaussianFourierProjection`（`head.py:46-57`） |
| 数据集注册 | `train.py:41-108` | 新建 `data_register.py`，路径 CLI 化 |
| 预训练权重 | — | **暂不使用** `petrain/diffdet_coco_res50.pth`（COCO 80 类，head cls 层形状不匹配；若要用必须 `strict=False` 且丢弃 cls 层，列为可选 A10） |

---

## 7. 风险登记表

| ID | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | **Box Renewal 破坏多步历史**（每步裁掉低分框再随机补齐，缓存的 `x_i/ε_i` 不再同分布） | 高：多步求解器理论失效 | 主矩阵关闭 renewal；单独 A1 试验三种策略：① 全关 ② renewal 后降阶重启 ③ per-slot mask 混合更新 |
| R2 | proposal 数量在步间变化 | head 输入 shape 抖动 | shape 断言 + 更新时按当前 `img.shape` 广播 |
| R3 | head 每步同时输出 cls，而 cls 不参与 ODE | 历史缓存只需存 box | `denoise_fn` 只返回 box；eval 时单独取最后一步 cls |
| R4 | torch 1.10 无 `forward_ad` | EMS 算不了 | 有限差分 JVP（精度足够，见 M3 Gate 的抖动验收） |
| R5 | EMS 统计量与数据 shape/Proposal 数绑定 | 不可复用 | 采用 `per_coord` `(N+1,1,4)` 与各向同性假设（§3.4） |
| R6 | Windows 上 detectron2 0.6 + torch 1.10 组合 | 潜在 DLL/编译问题 | 环境已实测可用；`NUM_WORKERS` 先设 0，若稳定再调 2 |
| R7 | 6GB 显存，IMS_PER_BATCH 只能 2 | 训练慢、BN 统计差 | AMP + Freeze backbone stage1（可选 A11）+ 用 `SyncBN` 无意义故保持 `FrozenBN` 策略 |
| R8 | 现有 `detector_dpm3.py` 半成品误导 | 浪费时间 | **明确不作为基线**，仅参考；M1 重写 |
| R9 | `_background_` 类别污染 | AP 与 loss 全错 | `data_register.py` 显式 id_map（已定位） |
| R10 | Ensemble 混淆"多步更好"的来源 | 结论不可信 | FEP §2.3 强制解耦 |
| R11 | EDM `λ(σ)` 与 Hungarian set-loss 不兼容 | 训练不稳 | 只作用于 reg 分支（A4 验证） |
| R12 | Beta′ 采样开销（NumPy 在 Windows worker） | dataloader 变慢 | 预生成 1e6 查表循环 |
| R13 | PFGM++ `N=4P=1200` 时 Beta′ 过于集中，`D` 无效 | H2 不成立 | 扫足够宽的 `D`；必要时改 `per_proposal` 粒度（A7），使 `N=4` 让重尾效应显著 |

---

## 8. 目录与文件约定

```
fast-diffusiondet/
├── docs/
│   ├── BLUEPRINT.md        ← 本文件（只增不改，重大变更升版本）
│   ├── PROGRESS.md         ← 实时工作状态（每次动手都要更新）
│   └── RESULTS.md          ← 实验数据（M2 起）
├── configs/lab/            ← 全部新增配置（不动原 configs）
├── diffusiondet/solvers/   ← 新增采样器包
├── statistics/<dataset>/<tag>/{l.npz,sb.npz}
├── scripts/
│   ├── calib_sigma_data.py
│   ├── bench_latency.py
│   └── run_matrix.ps1
└── results/raw/*.csv       ← 机器可读原始结果（唯一数据源）
```

**命名规范**：`<dataset>.<formulation>.<solver>.<nfe>N.se<seed>`，例 `sdd.f0.dpv3.nfe4.se42`。

---

## 9. 工作协议（"实时记录工作状态"怎么落地）

1. 每次开始动手前：`todo_write` 标记对应里程碑 `in_progress`，并在 `PROGRESS.md` 追加一行
   `| YYYY-MM-DD HH:MM | <正在做的事> | 进<br>行中 |`。
2. 每完成一个 Gate 条目：`RESULTS.md` 记录数据 → `PROGRESS.md` 勾选该 Gate → `todo_write` 更新状态。
3. 遇到与蓝图不符的事实：**先更新蓝图**（标注 `v1.x 修订`），再改代码；禁止"代码已变、文档没变"。
4. 每个 hypothesis（H1–H4）在 `RESULTS.md` 中常驻一行"当前判定"，随数据更新。

---

## 附：参考位置速查

| 需要的文件 | 绝对路径 |
|---|---|
| DiffusionDet 源码 | `e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet/` |
| DDIM 主循环 | `e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet/diffusiondet/detector.py` (186-274) |
| EDM 采样器 | `e:/Files/DL-code/model+/diffusion-model/edm-main/generate.py` (25-60 heun, 66-176 ablation) |
| EDM 预条件 | `e:/Files/DL-code/model+/diffusion-model/edm-main/training/networks.py` (660-663) |
| EDM 损失 | `e:/Files/DL-code/model+/diffusion-model/edm-main/training/loss.py` (65-80) |
| DPM-Solver-v3 | `e:/Files/DL-code/model+/diffusion-model/DPM-Solver-v3-main/dpm_solver_v3.py` (740-793 sample) |
| EMS 计算参考 | `e:/Files/DL-code/model+/diffusion-model/DPM-Solver-v3-main/compute_EMS_scoresde.py` (≈194) |
| PFGM++ 损失 | `e:/Files/DL-code/model+/diffusion-model/pfgmpp-main/training/loss.py` (111-176) |
| PFGM++ 采样器 | `e:/Files/DL-code/model+/diffusion-model/pfgmpp-main/generate.py` (28-70, 203-227) |
| PLS 数据 | `e:/Files/DL-code/model+/diffusion-model/dataset/PLS/Plantv2` |
| SDD 数据 | `e:/Files/DL-code/model+/diffusion-model/dataset/SDD/Strawberry` |
