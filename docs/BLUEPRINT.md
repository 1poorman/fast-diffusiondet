# DiffusionDet 高效扩散机制对比实验 —— 技术蓝图

> 目标：把 `fast-diffusiondet` 的 DDIM 推理替换为更高效的采样机制，系统性地对比
> **DDIM（基线） / EDM（Karras + Heun） / PFGM++（重尾泊松流扰动） / DPM-Solver-v3（EMS 高阶多步）**
> 四种机制在**同等 NFE 预算**下的检测精度与推理耗时，并给出可复现的实验结论。
>
> 文档版本：**v2.0** ／ 2026-09-25 硬件适配重写（基线：v1.0，2026-09-23，Windows/3060 端撰写）
> 实时进度见同目录 `PROGRESS.md`，实验数据见 `RESULTS.md`

---

## 修订记录

| 版本 | 日期 | 变更摘要 |
|---|---|---|
| v1.0 | 2026-09-23 | 初版。基于 Windows + RTX 3060 6GB 单卡勘查撰写，全部绝对路径为 Windows 端 |
| **v2.0** | **2026-09-25** | **硬件适配重写**（决策 D3/D4）：① 环境改为 Linux 工作站 4×RTX 3090 24GB（共享）；② 全部路径换成本机实测值；③ 训练策略升级：bs 2→8、LR 线性放大、arm-per-GPU 并行调度、新增 E0-control 对照臂；④ 解除 torch 1.10 无 `forward_ad` 约束（R4 消除），EMS 改 forward-AD JVP 主路线；⑤ M3–M6 里程碑与排期按新硬件重估；⑥ 风险表更新（删 R4/R6，新增 R14–R16）；⑦ 记录 M0–M2 已完成事实（Linux 端） |

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
   会**破坏多步求解器依赖的历史缓存与 ODE 轨迹语义**。已在 M2 验证：主矩阵关闭 renewal/ensemble，
   renewal/ensemble 仅作 A1/A2 消融（M2 发现 17：它们是"修补"，ODE 求解器是"根治"）。
4. **本机硬件：Linux 工作站，4× RTX 3090 24GB，但为共享机器**（实测 GPU 2/3 被他人进程占满 24GB，
   GPU 0 部分占用）。**常态可靠可用 1–2 卡，空闲时最多 4 卡**。所有实验按"arm-per-GPU 并行、
   宽度 1–2 保守排期"设计（D4），启动前必须查 `nvidia-smi` 并记录快照。
5. **M0–M2 已完成**（Linux 端，2026-09-24）：baseline 本机重训 **AP=61.83**；DDIM 逐 bit 等价；
   收敛阶 0.90/2.23/2.33；**H1 已判定支持**（DPMv3-degenerated@NFE=4 比 DDIM@4 **+1.31 AP**）。
   剩余 M3（EMS）→ M6（确认与报告）。

---

## 1. 现状勘查（已实测，非推测）

### 1.1 运行环境（2026-09-25 实测，替换 v1.0 全部 Windows 条目）

| 项 | 实测值 | 备注 |
|---|---|---|
| OS | **Linux**（共享工作站） | detectron2 官方支持平台 |
| 仓库路径 | `/home/huachenghao/codes/fast-diffusiondet` | 下文 `<repo>` |
| conda env | **`fastdiff`** @ `/home/huachenghao/.conda/envs/fastdiff` | |
| Python | **3.10.21** | |
| PyTorch | **2.1.2 + cu121**，4 GPU 可见 | ⚠ 与 v1.0 的 1.10 不同——见下方 R4 解除 |
| detectron2 | **0.6**（源码编译） | |
| numpy | 1.26.4 | |
| GPU | **4× NVIDIA RTX 3090，24 GB/卡** | **共享**：实测 GPU2/3 被他人占满（24 GB），GPU0 部分占用（~5 GB），**可靠空闲 ≈ 1–2 卡** |
| 训练吞吐 | **0.30 s/it @ bs=2**（SDD，3090） | v1.0 3060 上为 0.50 s/it；24 GB 显存下 bs 上限远未到 |
| 单卡 eval 延迟 | NFE=1 **37.6 ms**（eval batch=8，中位数） | 见 RESULTS §4.2 |

> ✅ **R4 解除（v2.0）**：torch 2.1.2 已有 `torch.autograd.forward_ad` 且 `torch.func.jvp` 实测可用。
> M3 的 EMS 统计量改用 **forward-AD JVP 为主实现**（精确、比有限差分少一半 head 调用），
> 有限差分 JVP 降级为交叉校验工具（见 §3.4、D5）。

> ⚠ **跨硬件可比性（R16）**：M0 的 3060 数字（AP=60.48、latency 108.5 ms）与 M2 的本机数字
> （AP=61.83、latency 37.6 ms）**不可跨表比较**（torch 版本 + 硬件差异，M2 发现 19）。
> 所有 M2+ 结论以本机 ckpt 为基准；3060 数据仅作历史存档。

### 1.2 可用数据（实测）

| 数据集 | 本机路径 | train | val | thing 类数 | 备注 |
|---|---|---|---|---|---|
| **PLS / Plantv2** | `<repo>/datasets/PLS/Plantv2` | 7916 img / 7915 ann | 2024 img / 2024 ann | **16** | 另有 test2017 **111 张且带标注**（`instances_test2017.json`）→ M6 留作 held-out 终测 |
| **SDD / Strawberry** | `<repo>/datasets/SDD/Strawberry` | 1750 img / 3932 ann | 750 img / 1775 ann | **7** | 实例较密，阶段 A 筛选集 |
| HBueHxOW / wheat_seg | `<repo>/datasets/HBueHxOW` | — | — | — | 备选第三数据集（A13，可选，跨数据集稳定性检查） |

> 上表路径中 `SDD`、`PLS` 为仓库内符号链接（指向同目录下 `Strawberry`、`Plantv2` 真实目录），
> 注册代码用 `datasets/SDD|PLS` 即可（M2 已验证通过）。
> 数据格式均为 **COCO json**，目录结构 `train2017/ val2017/ annotations/instances_{train,val}2017.json`。

> ⚠ **类别 id 陷阱（v1.0 已实测，仍然有效）**：两个 json 的 `categories` 都含 `(0, '_background_')`，
> 但 `annotations` 的 `category_id` **从不等于 0**。
> → 必须**显式构造** `thing_dataset_id_to_contiguous_id = {cid: cid-1 for cid in 1..K}`，
> 设置 `thing_classes` 为 K 个真实类名，**`NUM_CLASSES = K`**（PLS=16, SDD=7）。
> 已在 M0 落地（`data_register.py` + `datasets/.cache_fastdd/*.clean.json` 剔除 background 的评测 json，
> M0 发现 8②：COCOEvaluator 直接用 json 重建类别，评测 split 必须用干净 json）。

### 1.3 fast-diffusiondet 关键机制定位

**包路径**：`<repo>/diffusiondet/`

| 机制 | 文件:行 | 现状 |
|---|---|---|
| 噪声调度 `cosine_beta_schedule` | `detector.py:49-59` | T=1000，`s=0.008` |
| 扩散超参初始化 | `detector.py:83-128` | `objective='pred_x0'`、`ddim_sampling_eta=1.`、`self_condition=False`、`scale=SNR_SCALE=2.0` |
| GT 加噪 `prepare_diffusion_concat` | `detector.py:370-405` | GT + `randn/6+0.5` 占位补齐至 `num_proposals` |
| 前向加噪 `q_sample` | `detector.py:277-284` | 标准 VP：`x_t = √ᾱ_t·x0 + √(1-ᾱ_t)·ε` |
| **模型输出转换 `model_predictions`** | `detector.py:170-184` | **x0 参数化**：head 输出绝对坐标 → 归一化 → `(x*2-1)*scale`；再由 `predict_noise_from_start` 反算 ε |
| **DDIM 采样主循环** | `detector.py:186-274`（现为冻结锚点） | M1 已抽至 `solvers/ddim.py`，逐 bit 等价（RESULTS §3.5） |
| Box renewal | `detector.py:209-219, 236-238` | 分数>0.5 裁剪 + `randn` 补齐；已有 cfg 开关 `BOX_RENEWAL` |
| Ensemble + NMS | `detector.py:239-274` | 已有 cfg 开关 `USE_ENSEMBLE`；仅当 `sampling_timesteps > 1` 时生效 |
| 推理入口 | `detector.py:313-315` | **backbone features 只算一次**，NFE = head forward 次数 |
| 时间步嵌入 | `head.py:31-43, 96-101, 271-274` | `SinusoidalPositionEmbeddings(256)` → MLP(1024) → FiLM |
| 配置项 | `config.py:11-77` | `SNR_SCALE=2.0`、`SAMPLE_STEP=1`、`NUM_PROPOSALS=300`、`USE_NMS=True` |
| 损失 | `loss.py` | `HungarianMatcherDynamicK` + `SetCriterionDynamicK`：Focal + L1 + GIoU，**不是 ε-MSE** |
| **求解器包（M1 新建）** | `diffusiondet/solvers/` | `base.py` / `schedule.py` / `ddim.py` / `heun.py`（含 `HEUN_MAX_DW` 步长守卫）/ `dpm_solver_v3.py` |

**M2 已实测的 solver 行为（写死在这里作为后续设计的边界条件）**：

- `DDIM(eta=1) 随 NFE 单调退化`（61.98@1 → 57.63@10，M2 发现 16）→ 纯 ODE（eta=0）路径全面占优，
  E0 主矩阵比较对象是 DDIM(eta=1)（原仓库默认行为），报告中必须注明 eta 设置。
- `Heun 二阶校正在 cosine 离散调度首步 |Δw|≈2e4 处灾难性发散`（M2 发现 15）→ 已用
  `HEUN_MAX_DW=1.0` 步长守卫修复（多阶 ramp-up 标准做法）；M4 的 EDM 重训臂因 σ 空间不同需重新验证守卫。
- `DPMv3 同 NFE 比 DDIM 慢 50–80%`（M2 发现 18）→ 工程开销（逐调用插值查表 + 每次重建 solver），
  M3 任务：预构建 + 缓存。

### 1.4 坑清理状态（v1.0 §1.4 的后续）

| # | v1.0 严重度 | 问题 | 现状（v2.0） |
|---|---|---|---|
| B1–B3 | 🔴 | `detector_dpm3.py` 必崩三处 | ✅ 已归档 `legacy/experimental/`，M1 从零重写 `solvers/` |
| B4 | 🟡 | CIFAR-10 版 `l.npz/sb.npz` 不可用 | ✅ 随 legacy 归档；M3 自算 box 版 EMS |
| B5 | 🟡 | `samplers/` 缺 `__init__.py` | ✅ M1 新包 `solvers/` 已带 |
| B6 | 🟡 | 三处同名注册 `"DiffusionDet"` | ✅ legacy 已解除 `__init__.py` 导出 |
| B7 | 🟠 | `train.py` 硬编码 WGISD 路径 | ✅ 原 `train.py` 已归档，统一用根目录 `train_net.py` + CLI |
| B8 | 🟠 | `configs/diffdet.coco.res50.yaml` 名不副实 | ✅ 新建 `configs/lab/`，原 configs 不动 |
| B9 | 🟡 | 孤立 `Base-DPM3Det.yaml` | ⬜ M3 起接管或删除（仍挂起） |

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
| **成本（主）** | `NFE` | head forward 次数，代码内计数器取数而非推算 |
| **成本（主）** | `latency_ms` | 单图 batch=1 另行专项 bench；eval 吞吐用于矩阵内相对比较，两者**不得混用**（M2 起 eval 用 batch=8 计时，见 RESULTS §4.2 注） |
| **成本（辅）** | `peak_mem_GB` | `torch.cuda.max_memory_allocated()` |
| **训练成本** | `train_wallclock_h`, `it/s`, `img/s` | M0 已有基线；M4 起每次 arm 记录 |

### 2.3 公平对比协议（FEP —— Fair Evaluation Protocol，极其重要）

不遵守这条，所有数字都不可信：

1. **同 checkpoint 对比**：training-free 系列（M2/M3）必须加载**同一个** baseline checkpoint，权重完全冻结。
2. **解耦 ensemble / renewal**：主矩阵在 `BOX_RENEWAL=False + USE_ENSEMBLE=False` 下跑，得到**纯采样器的 AP-NFE 曲线**；ensemble 与 box renewal 单独作为 ablation（A1/A2）。
3. **固定随机初值**：同一 NFE 下不同 solver 使用**同一个 `torch.randn` 种子**（可注入 `noise` 参数），消除 Monte-Carlo 噪声；每个配置跑 3 seed 取均值±std（D6：保留 3 seed，eval 在本机已很便宜）。
4. **同 NMS / 同 proposal 数**：`USE_NMS=True`、阈值 0.5、`NUM_PROPOSALS` 全矩阵一致。
5. **训练侧等预算**：所有需要重训的 arm 使用**相同的 MAX_ITER / batch / LR schedule / seed / 数据增广**；预算以 **epoch 数**（img-passes）定义，不以 iter 数定义（v2.0 强调，因为 bs 从 2 变 8）。
6. **数字只记一处**：所有结果实时写入 `RESULTS.md`，脚本自动 append CSV（`results/raw/*.csv`），禁止手工抄写到多处。
7. **（v2.0 新增）控制臂原则**：凡训练范式变更（M4/M5），必须同步重训同预算的 **F0 控制臂**（E0-DDIM@新预算）作为对照，禁止拿旧预算 baseline 比新预算 arm。
8. **（v2.0 新增）共享机器纪律**：每个训练/eval 任务启动前 `nvidia-smi` 快照记入 `PROGRESS.md` 工作日志；用 `CUDA_VISIBLE_DEVICES` 绑卡；单任务显存上限 20 GB（给他人留余量）；禁止默认占满 4 卡。

### 2.4 两阶段实验协议（D1 维持，预算升级）

- **阶段 A（筛选）**：**SDD/Strawberry**，新预算 **8 epoch @ bs=8 ≈ 1750 iter**（v1.0 为 4 epoch @ bs=2；
  img-passes ×2，但 3090 上单臂仅 ~20 min，全矩阵成本可忽略）。跑全 matrix + 全部消融。
- **阶段 B（确认）**：阶段 A 的 **top-2 组合 + E0 控制臂** 在 **PLS/Plantv2** 上完整训练
  （**12k iter @ bs=8 ≈ 12 epoch**，~2 h/arm）。PLS test2017（111 张带标注）留作 held-out 终测（M6）。

> ✅ **D1 维持理由（v2.0 复核）**：硬件变快并不改变"先在 1750 张图上筛选、再到 7916 张图上确认"的
> 统计逻辑——若直接在全矩阵上跑 PLS，arm 选择会过拟合 PLS。两阶段的成本现在已可忽略，收益不变。
> 唯一变化：阶段 A 预算从 4 epoch 提到 8 epoch（降低短周期训练噪声，M0 发现 10 表明
> LR 衰减点位置对短周期 AP 影响极大，预算太低时 arm 间差异会被调度噪声淹没）。

### 2.5 训练策略（v2.0 新增，决策 D4）

| 项 | v1.0（3060 6GB） | v2.0（3090 24GB） | 理由 |
|---|---|---|---|
| `IMS_PER_BATCH` | 2 | **8** | 24 GB 显存（实测 bs=2 仅用 3.3 GB 是 3060 数据；bs=8 + AMP 预计 <12 GB，仍留余量） |
| `BASE_LR` | 2.5e-5 | **1e-4**（×4，线性规则） | 配合 bs ×4；短周期下调度过敏感（M0 发现 10），所有 arm 必须同 STEPS |
| `NUM_WORKERS` | 0（Windows 排障） | **4** | Linux 无 d2 worker 问题 |
| 并行方式 | 串行单卡 | **arm-per-GPU 并发**（宽度 1–2，空闲时 4） | 共享机器，不做单机多卡 DDP（FrozenBN 无 BN 同步收益，且多卡合一任务降低调度灵活性） |
| 预算表达 | iter 数 | **epoch 数（img-passes）** | FEP #5；换 bs 后 iter 数不可比 |

> ⚠ **等 img-passes ≠ 等收敛**：bs 变大后每 epoch 的更新步数变少，8 epoch@bs=8 的梯度更新次数
> = 4 epoch@bs=2。**因此 E0 控制臂（F0@bs=8 重训）是绝对必须的**（FEP #7）——它同时吸收了
> bs/LR/调度变化的影响，使 F1/F2 的增益可归因于训练范式本身。

---

## 3. 技术方案

### 3.1 统一抽象层（M1 产出，已完成）

包 `diffusiondet/solvers/`（已带 `__init__.py`）：

```
solvers/
├── __init__.py              # SAMPLER_REGISTRY + build_solver(cfg)
├── base.py                  # DenoiseFn 协议 + BoxSolver 基类 + NFE 计数器
├── schedule.py              # VPSchedule：α(t),σ(t),λ(t),λ⁻¹（w=σ/α 参数化）
├── ddim.py                  # 逐 bit 等价（RESULTS §3.5 Gate 1 验证）
├── heun.py                  # w=σ/α 上 Euler/Heun，HEUN_MAX_DW 步长守卫
├── dpm_solver_v3.py         # 移植自 thu-ml/DPM-Solver-v3，统计量 1 维化、去 .cuda 硬编码
└── statistics.py            #（M3）EMS 统计量计算
```

**唯一的去噪函数接口**：`denoise_fn(x, sigma_or_t) -> x0_hat`（`(B,P,4) -> (B,P,4)`），
内部完成 clamp→反归一化→xyxy→乘 whwh→`head(...)`→反变换。除 DDIM 外所有 solver "看不见"
`images_whwh` / `backbone_feats` / `batched_inputs`，三个 solver 可互相对拍。

### 3.2 基线 S0：DDIM（已完成，冻结锚点）

`detector.py` 原 `ddim_sample` 保留为冻结锚点，`solvers/ddim.py` 逐 bit 等价已验证（max|Δ|=0）。

### 3.3 机制 A：EDM

**(A-1) 采样器**（training-free）—— M2 已用更直接的方案落地（M1 发现 14）：
**直接在离散 cosine VP 调度上积分 PF-ODE**（heun.py 的 w=σ/α 参数化），
时间网格与 DDIM 完全同一份，保证同 NFE 公平对比；不引入 ablation_sampler 的线性 VP 拟合误差。
`NFE = 2N−1`（Heun）/ `N`（Euler）；`HEUN_MAX_DW` 守卫已在 M2 验证。

**(A-2) 训练范式**（需重训，M4 核心）—— 参考上游 `NVlabs/edm`（本机无副本，M4 时 clone）：

```
训练: σ ~ LogNormal(P_mean=-1.2, P_std=1.2);  x = x0 + σ·ε
预条件: c_skip = σ_data²/(σ²+σ_data²)
        c_out  = σ·σ_data/√(σ²+σ_data²)
        c_in   = 1/√(σ_data²+σ²)
        c_noise= ln(σ)/4
前向:  D(x;σ) = c_skip·x + c_out·F(c_in·x, c_noise)
损失加权: λ(σ) = (σ²+σ_data²)/(σ·σ_data)²
```

**落到 DiffusionDet 的映射困难点（必须设计后验证，与 v1.0 一致）**：
- loss 是匈牙利匹配后的 `Focal + L1 + GIoU` → `λ(σ)` **只作用于 regression 分支**（L1+GIoU），
  classification（Focal）保持原样（A4 验证）。
- `σ_data` 实测：统计训练集 `prepare_diffusion_concat` 输出 `x_start` 的逐坐标标准差，
  一次性脚本 `scripts/calib_sigma_data.py` 产出 `sigma_data.json`。
- `c_in·x` 与 `SNR_SCALE` 语义重复，二者只能取一（A5）。
- 时间嵌入：`c_noise=ln σ/4 ∈ [-1.55, 1.10]` 输入域剧变 → A6：`Sinusoidal` vs
  `GaussianFourierProjection`，及缩放因子。

> ⚠ **v2.0 新增**：M4 的 EDM arm 训练噪声在 σ 空间，推理时求解器的时间网格也要从 cosine VP
> 网格换成 EDM/Karras 网格——E1-HEUN 用 Karras ρ=7，E1-DPV3 的 `NoiseScheduleVP` 需换
> `EDM` schedule（原 DPM-Solver-v3 支持，移植版已保留分支）。守卫 `HEUN_MAX_DW` 在
> Karras 网格下需重新标定（首步 |Δw| 行为不同）。

### 3.4 机制 B：DPM-Solver-v3（来自 `thu-ml/DPM-Solver-v3`，本机无副本，M3 时 clone）

DiffusionDet 的 VP 前向与 `NoiseScheduleVP` 假设完全吻合（`α²+σ²=1`），
`objective='pred_x0'` 对应 `model_type="x_start"`。调用骨架（M1 已移植）：

```python
dpm = DPM_Solver_v3(statistics_dir=..., noise_schedule=ns, steps=k,
                    skip_type="logSNR", degenerated=False, device="cuda")
x0 = dpm.sample(x_T, model_fn, order=3, p_pseudo=False, use_corrector=True,
                c_pseudo=True, lower_order_final=True)
```

**EMS 统计量的本项目适配（M3 核心，v2.0 更新）**：

原版对 batch 维取均值得到逐像素统计量。本项目 `x ∈ (B,P,4)`，proposal 之间**可交换**，
允许降维（R5）：

| 模式 | 统计量形状 | 说明 |
|---|---|---|
| `scalar` | `(N+1, 1, 1)` | 所有坐标共用 |
| `per_coord` | `(N+1, 1, 4)` | cx/cy/w/h 各自统计（**推荐**） |
| `per_prop` | `(N+1, P, 4)` | 原版等价，**不推荐**（与 renewal 冲突、无收益） |

**JVP 实现（v2.0 主路线 = forward-AD，D5）**：torch 2.1.2 的 `torch.func.jvp` 实测可用，
对每个 logSNR 网格点 `i`、随机方向 `v ~ N(0,I)`：

```
l_i = mean_over_batch_and_coords( σ_i · JVP(D, x, v) )      # 一次前向 + 前向微分
```

`denoise_fn` 内部含 clamp（不可微分边界外推区域）——forward-AD 在 clamp 的活跃区给出精确导数，
与有限差分相比**无截断误差**；有限差分 JVP（`eps_fd ≈ (D(x+h·v) − D(x−h·v))/2h`）保留为
`statistics.py` 的 `--method fd` 交叉校验模式。新增 Gate（M3）：两种方法在抽样网格点上
`l` 的相对偏差 `< 5%`。

再按原版方式积分出 `s, b`（沿用 `weighted_cumsumexp_trapezoid`，移植版已 import），
产出 `statistics/<dataset>/<tag>/{l.npz,sb.npz}`。

**网格并行（v2.0）**：1000 个 logSNR 网格点按卡切分（`CUDA_VISIBLE_DEVICES` 各跑一段），
结果拼接；单卡预算内（Gate：总 wall-clock < 1 h，较 v1.0 的 2 h 收紧）。

**`degenerated=True` 是关键对照组**（等价 DPM-Solver++，不需真实 EMS）：
M2 已用它出结果（E0-DPP），M3 把真统计量的增量收益单独量化（H3）。

**DPMv3 latency 优化（M3 必办，M2 发现 18）**：预构建 `NoiseScheduleVP` 插值表（λ 网格一次性
向量化）、solver 实例按 `(steps, order, skip_type)` 缓存复用、marginals 常驻 GPU。
目标：同 NFE latency 与 DDIM 差 < 15%（M2 Gate 唯一未过项）。

### 3.5 机制 C：PFGM++（来自 `NVlabs/pfgmpp`，本机无副本，M5 时 clone）

**已核实**：PFGM++ 相对 EDM 只有两处不同——训练扰动核（Beta′(N/2, D/2) 重尾径向 × 均匀方向）
+ 采样先验；网络/preconditioning/σ 分布与 EDM 完全一致。**PFGM++ = EDM + 两行扰动核改动**，
在 M4 的 EDM 基础上叠加，性价比高。

**落到 box 空间的映射**：
- 数据维数 `N = 4 × P`（P=300 → N=1200），Beta′ 在 N≥1200 时极度集中在均值附近 →
  扫 **D ∈ {32, 128, 512, 2048, ∞(即纯 EDM)}**，阶段 A 选。
- radial 归一化粒度 ablation A7：`global`（整图所有 proposal 展平成 N 维向量，忠于原论文）vs
  `per_proposal`（每个 proposal 4 维单独 Beta′，贴合 DiffusionDet 的 proposal 独立性）。
  R13 对策：若 global 粒度下重尾效应不显著（H2 无差异），per_proposal 使 N=4 让重尾效应显著。
- `pfgmpp-target` 泊松场经验目标默认 `stf=False` 不启用，**不要碰**。

> ℹ **v2.0**：v1.0 的"NumPy beta 在 Windows worker 成瓶颈"（R12）随平台迁移自然消解；
> 预生成查表（1e6 个）仍保留为可选项（Linux 下 `np.random.beta` 每 step 开销可忽略）。

### 3.6 三者的组合关系

```
训练侧 ──► 扰动核:    高斯(EDM/DDIM)   or   Beta′ 重尾(PFGM++, D=超参)
      └─► 参数化:    VP-DDPM                      or   EDM c_skip/c_in/c_out/c_noise
      └─► 加权:      无 / EDM λ(σ)（作用于 reg 分支）
推理侧 ──► 求解器:    DDIM(1阶)  Heun(2阶)  DPMv3-degenerated(≈DPM-Solver++)  DPMv3-EMS(高阶+EMS)
```

**研究假设（H1 已判定，其余待验证，写死在这里以便事后对照）**：
- ~~H1~~：**✅ 已支持（M2）**：NFE≤4 时 DPMv3(DPP) 显著优于 DDIM（@4: +1.31 AP，@2: +0.55 AP）。
- H2：PFGM++ 的直线化轨迹使 2 步即可接近 EDM 8 步的 AP（≥ −0.5 AP 内）。（M5；比较基准改为同预算 EDM 臂）
- H3：EMS 真统计量相对 `degenerated` 仅在 NFE ≤ 5 时有可见收益（≥ +0.3 AP）。（M3）
- H4：EDM 参数化主要提升**上界**（高 NFE 的 AP），而非低 NFE 效率。（M4；判定基准 = 同预算 E0 控制臂）

---

## 4. 实验矩阵

编号规则：`E{训练范式}-{采样器}`，NFE 一律取值 `{1,2,3,4,6,8,10}`。

| ID | 训练范式 | 采样器 | 训练成本（v2.0 实测估） | 说明 |
|---|---|---|---|---|
| **E0-DDIM** | F0 VP-DDPM（原） | DDIM(eta=1) | **~20 min/arm @ SDD** | **基线曲线**；M4 起需重训同预算控制臂 |
| **E0-CTRL** | F0（同上） | —（不评测） | ~20 min | **v2.0 新增控制臂**：F0@bs8/8epoch，M4/M5 全部 arm 的判定基准（FEP #7） |
| E0-HEUN | F0 | Heun | 0 | training-free，M2 已完成 |
| E0-EULER | F0 | Euler | 0 | 剥离"阶数"与"时间步"收益，M2 已完成 |
| E0-DPP | F0 | DPMv3 `degenerated=True` | 0 | ≈ DPM-Solver++，M2 已完成（H1 主力） |
| **E0-DPV3** | F0 | DPMv3 `degenerated=False` + EMS | 0 | **M3 主力** |
| E1-HEUN | F1 EDM 参数化 | Heun + Karras ρ=7 | ~20 min | M4 |
| E1-DPV3 | F1 | DPMv3 + EMS | ~20 min | M4（EDM σ 空间，schedule 换 EDM 分支） |
| E2-HEUN | F2 PFGM++ (D*) | Heun + 重尾先验 | ~20 min | M5，**预期最优** |
| E2-DPV3 | F2 | DPMv3 + EMS(per_coord) | ~20 min | M5/M6 组合王牌 |

**Ablation（编号 A*）**

| ID | 内容 |
|---|---|
| A1 | Box Renewal 开/关 × 多步求解器 ✅（M2 已完成，DDIM 上） |
| A2 | Ensemble+NMS 开/关 ✅（M2 已完成，DDIM 上） |
| A3 | `SNR_SCALE ∈ {1,2,4}` 与新参数化的交互 |
| A4 | EDM `λ(σ)` 只作用于 reg 分支 vs 全 loss |
| A5 | `SNR_SCALE` 与 `c_in/c_out` 二选一 |
| A6 | 时间嵌入：Sinusoidal vs GaussianFourier，及 `c_noise` 缩放因子 |
| A7 | PFGM++ radial 粒度：global vs per_proposal |
| A8 | EMS 统计量模式：scalar vs per_coord vs none(degenerated) |
| A9 | Karras `ρ ∈ {3,7,12}` 与 `skip_type ∈ {logSNR, time_uniform}` |
| A10 | 迁移初始化（COCO ckpt）vs 仅 ImageNet backbone 从头训练（D2 副作用对照，必做） |
| A11 | `FREEZE_BACKBONE_STAGE1`（现在仅作显存选项，24GB 下大概率不需要） |
| A13 | （v2.0 新增，可选）HBueHxOW/wheat 第三数据集跨数据集稳定性抽查 |

> A12 编号预留给"bs/LR 缩放本身的影响"（若 E0-CTRL 与旧 baseline 差异 >1 AP 才需要展开，
> 否则不跑，仅记录差异值）。

---

## 5. 里程碑与验收标准（Gate）

> 每个 Gate **必须全部满足**才能进入下一里程碑。Gate 检查表打勾在 `PROGRESS.md`。
> M0–M2 已完成（Linux 端，2026-09-24），其 Gate 结果见 `RESULTS.md` §3/§4，不再重复。

### M3 — EMS 统计量标定与 DPM-Solver-v3 全功率

**任务**
1. `statistics.py`：forward-AD JVP 主实现 + 有限差分交叉校验（`--method fd`），
   `scalar` 与 `per_coord` 两版，1000 logSNR 网格 × K=256，按卡切分并行。
2. DPMv3 latency 优化：预构建插值表、solver 缓存、marginals 常驻 GPU，重测 §4.2。
3. 跑 E0-DPV3 × NFE{1,2,3,4,6,8,10} × 3 seed。

**Gate M3**
- [ ] EMS 计算总 wall-clock < 1 h（4 卡并行切网格；单卡等价 < 2 h）
- [ ] forward-AD 与有限差分两条路线抽样点 `l` 相对偏差 < 5%
- [ ] `l/s/b` 无 NaN/Inf，`l` 随 t 单调光滑（相邻网格相对变化 < 10）
- [ ] **主验收（H3）**：`degenerated=False` 相对 `degenerated=True` 在至少一个 NFE ≤ 5 的点上 **AP 提升 ≥ 0.3**
- [ ] DPMv3 同 NFE latency 与 DDIM 差 < 15%（补齐 M2 唯一未过 Gate）
- [ ] 完成 A8（三种统计量模式）消融

### M4 — EDM 训练范式移植

**任务**：引入 `c_skip/c_in/c_out/c_noise`、`σ~LogNormal(P_mean,P_std)`、`λ(σ)` 加权（仅 reg 分支）、
Karras ρ=7 时间步；`scripts/calib_sigma_data.py` 实测 `σ_data`；**同步重训 E0-CTRL 控制臂**；
arm 列表：E0-CTRL、E1-HEUN、E1-DPV3 + A4/A5/A6 各一档（共 ~6 arm，2 卡并行约 1.5 h wall-clock）。

**Gate M4**
- [ ] **E0-CTRL 已产出**且与旧 baseline（4 epoch@bs=2）差异已记录（预期 <1 AP；>1 AP 触发 A12）
- [ ] **等预算训练不退化**：E1 在 NFE=10 时的 AP ≥ E0-CTRL − 0.5
- [ ] `σ_data`、`P_mean`、`P_std` 的标定过程与数值写入 `RESULTS.md`
- [ ] 完成 A4、A5、A6 三组消融
- [ ] **主验收（H4）**：E1 的高 NFE AP 上界相对 E0-CTRL 提升 ≥ 0.5 AP，**或**明确否定 H4（否定也是有效结论）

### M5 — PFGM++ 训练范式移植

**任务**：在 M4 的 EDM 基础上加 Beta′ 重尾扰动核 + 重尾先验；扫 `D ∈ {32,128,512,2048,∞}`；
A7 粒度两档（共 ~7 arm，2 卡并行约 1.5 h wall-clock）。

**Gate M5**
- [ ] 训练 loss 稳定收敛（无 NaN，loss 曲线与 EDM 版同量级）
- [ ] **主验收（H2）**：存在某个 `D`，使得 **NFE=2 时的 AP ≥ E1@NFE=8 的 AP − 0.5**（基准为同预算 E1 臂，非旧 baseline）
- [ ] 完成 A7 粒度消融
- [ ] 单个 `D` 的完整训练 wall-clock 不超过 EDM 版本的 1.1×

### M6 — 组合、确认与报告

**任务**：阶段 A（SDD）选出 top-2 组合 + E0-CTRL，在 **PLS/Plantv2** 上 12k iter @ bs=8 完整训练
（~2 h/arm，2 卡并行 ~3.5 h wall-clock）；top 配置在 PLS **test2017**（held-out 111 张）终测；
出版 `RESULTS.md` 终稿。

**Gate M6**
- [ ] PLS 上的结论与 SDD **方向一致**（否则需在报告中明确讨论跨数据集不稳定性）
- [ ] 所有 hypothesis H1–H4 均有明确 "支持/不支持/部分支持" 结论及数据支撑
- [ ] 最终推荐配置给出 **AP vs NFE vs latency** 三维 Pareto 前沿表
- [ ] `RESULTS.md` 中每张表都能由 `results/raw/*.csv` 复现

**排期总览（v2.0，宽度 2 卡并行；宽度 1 时 wall-clock ×2）**

| 里程碑 | GPU 时长 | wall-clock | 备注 |
|---|---|---|---|
| M3 | ~2.5 h | ~1.5 h | EMS 并行 + latency 优化 + eval 矩阵 |
| M4 | ~2 h | ~1.5 h | ~6 arm |
| M5 | ~2.5 h | ~1.5 h | ~7 arm |
| M6 | ~6 h | ~3.5 h | PLS 3 arm × 2 h |
| **合计** | **≈ 13 h GPU** | **≈ 8 h** | 按 1.5× 调试余量 → 约 **12–15 h wall-clock** |

---

## 6. 改造点地图

| 目标 | 现有位置 | 计划改动 |
|---|---|---|
| 求解器包 | `diffusiondet/solvers/` | ✅ M1 完成 |
| 去噪函数提取 | `detector.py:170-184` | ✅ M1 完成（闭包 `make_denoise_fn`） |
| Box renewal / Ensemble 开关 | `detector.py:209-274` | ✅ M1 完成（`BOX_RENEWAL`/`USE_ENSEMBLE`） |
| 配置项 | `config.py:11-77` | 追加 `SOLVER/ORDER/SKIP_TYPE/DEGENERATED/STATS_DIR/BOX_RENEWAL/USE_ENSEMBLE` ✅；M4/M5 再追加 `FORMULATION/PFGM_D/P_MEAN/P_STD/SIGMA_DATA/TIME_EMB` |
| 训练加噪 | `detector.py:370-405` | 按 `FORMULATION` 分支：VP / EDM / PFGM++（M4/M5） |
| 前向扩散 | `detector.py:277-284` | 增加 EDM σ 形式分支（M4） |
| 损失加权 | `loss.py` | reg 分支乘 `λ(σ)`（EDM，M4） |
| 时间嵌入 | `head.py:96-101` | 接入 `c_noise`，可选 `GaussianFourierProjection`（M4） |
| 数据集注册 | `diffusiondet/data_register.py` | ✅ M0 完成（CLI 路径参数） |
| 迁移初始化 | `diffusiondet/weights.py` | ✅ M0 完成（`resolve_pretrain`） |
| E0-CTRL 配置 | `configs/lab/` | v2.0 新增 `sdd.res50.bs8.yaml`（bs=8、LR=1e-4、8 epoch），原 `sdd.res50.yaml` 保留作历史 |
| DPMv3 预构建缓存 | `solvers/dpm_solver_v3.py` | M3：插值表向量化 + solver 实例缓存 |
| EMS forward-AD | `scripts/compute_ems.py` + `solvers/statistics.py` | M3 主力 |

### 6.1 预训练权重加载协议（决策 D2，v2.0 路径更新）

**权重文件**：`diffdet_coco_res50.pth`（443 MB，COCO 80 类 DiffusionDet 完整权重）
**位置**：**不入库**（`.gitignore` 已排除 `*.pth`）。默认按序搜索：

1. `$FASTDD_PRETRAIN` 环境变量
2. `<repo>/pretained/diffdet_coco_res50.pth`（⚠ 目录名是 `pretained`，历史拼写，不改）
3. `<repo>/diffdet_coco_res50.pth`（仓库根，现有副本）

解析逻辑在 `diffusiondet/weights.py:resolve_pretrain(cfg)`，由 `train_net.py` 调用（M0 已实现）。

**加载方式**（`strict=False`，显式丢弃分类层）：backbone+FPN 直接加载；`head` 全部加载
（除 `cls_module` 末层 `[80,256]→[K,256]` 丢弃、按 `PRIOR_PROB=0.01` 重初始化）；
重初始化 seed 固定 `CLS_REINIT_SEED`。**F0/F1/F2 三范式必须加载同一份权重、同一 seed**（D2 公平性）。

> ⚠ **A10 消融仍然必做**（D2 副作用对照）：至少让 E0-CTRL 与 top-1 组合各跑一版
> "仅 ImageNet backbone、head 随机初始化"，确认结论不是 COCO 迁移带来的假象。

---

## 7. 风险登记表

| ID | 风险 | 影响 | 缓解 |
|---|---|---|---|
| **R1** | **Box Renewal 破坏多步历史** | 高：多步求解器理论失效 | ✅ 主矩阵已关闭 renewal；A1/A2 消融已完成（M2 发现 17） |
| R2 | proposal 数量在步间变化 | head 输入 shape 抖动 | shape 断言 + 按当前 `img.shape` 广播 |
| R3 | head 每步同时输出 cls，cls 不参与 ODE | 历史缓存只需存 box | `denoise_fn` 只返回 box；eval 单独取最后一步 cls |
| ~~R4~~ | ~~torch 1.10 无 `forward_ad`~~ | ~~EMS 算不了~~ | ✅ **v2.0 解除**：torch 2.1.2 有 forward-AD；有限差分降级为交叉校验 |
| R5 | EMS 统计量与数据 shape/Proposal 数绑定 | 不可复用 | `per_coord` `(N+1,1,4)` 与各向同性假设 |
| ~~R6~~ | ~~Windows 上 detectron2 0.6 + torch 1.10~~ | — | ✅ **v2.0 解除**：Linux 环境，M0–M2 已验证 |
| ~~R7~~ | ~~6GB 显存，bs 只能 2~~ | — | ✅ **v2.0 解除**：24GB，bs=8 |
| R8 | 现有 `detector_dpm3.py` 半成品误导 | 浪费时间 | ✅ 已归档 legacy，M1 已重写 |
| R9 | `_background_` 类别污染 | AP 与 loss 全错 | ✅ M0 已修（干净 json + 显式 id_map） |
| R10 | Ensemble 混淆"多步更好"的来源 | 结论不可信 | ✅ FEP §2.3 已解耦 |
| R11 | EDM `λ(σ)` 与 Hungarian set-loss 不兼容 | 训练不稳 | 只作用于 reg 分支（A4 验证） |
| R12 | Beta′ 采样开销 | dataloader 变慢 | ✅ Linux 下开销可忽略；查表方案保留为备选 |
| R13 | PFGM++ `N=4P=1200` 时 Beta′ 过于集中，`D` 无效 | H2 不成立 | 扫宽 D；必要时 `per_proposal` 粒度（A7） |
| **R14** | **共享工作站：他人进程抢占 GPU 或挤占显存**（实测 GPU2/3 已常被占满） | 任务 OOM 或排队 | FEP #8 纪律：`nvidia-smi` 快照 + `CUDA_VISIBLE_DEVICES` 绑卡 + 单任务 ≤20 GB + 排期按宽度 1–2 |
| **R15** | **bs 2→8 + LR ×4 在短周期下调度敏感**（M0 发现 10） | arm 间差异被调度噪声淹没 | 所有 arm 同 STEPS；E0-CTRL 必跑；首 200 it loss 与旧曲线对拍；异常则退回 bs=4/LR 5e-5 |
| **R16** | **跨硬件/跨 torch 版本数字不可比**（M2 发现 19） | 误用旧数字对比 | 全部结论以本机 ckpt（AP=61.83）为基准；3060 数据标记"历史存档" |

---

## 8. 目录与文件约定

```
fast-diffusiondet/
├── docs/
│   ├── BLUEPRINT.md        ← 本文件（重大变更升版本，见修订记录）
│   ├── PROGRESS.md         ← 实时工作状态（每次动手都要更新）
│   └── RESULTS.md          ← 实验数据（M2 起）
├── configs/lab/            ← 全部新增配置（不动原 configs）
│   ├── sdd.res50.yaml      ← M0 历史配置（bs=2，仅训练-free 评测继续用）
│   └── sdd.res50.bs8.yaml  ← v2.0 新训练配置（bs=8，M4 起所有训练臂）
├── diffusiondet/solvers/   ← 采样器包（M1）
├── statistics/<dataset>/<tag>/{l.npz,sb.npz}
├── scripts/
│   ├── calib_sigma_data.py
│   ├── compute_ems.py      ← M3，--method {jad,fd}，--shard 按卡切分
│   ├── bench_latency.py
│   └── run_m2_matrix.py    ← M2 已用，后续里程碑仿写
├── datasets/               ← SDD/PLS 符号链接 + .cache_fastdd 干净评测 json
├── pretained/diffdet_coco_res50.pth  ← 迁移初始化（gitignored）
└── results/raw/*.csv       ← 机器可读原始结果（唯一数据源）
```

**命名规范**：`<dataset>.<formulation>.<solver>.<nfe>N.se<seed>`，例 `sdd.f0.dpv3.nfe4.se42`。

---

## 9. 工作协议（"实时记录工作状态"怎么落地）

1. 每次开始动手前：`todo_write` 标记对应里程碑 `in_progress`，并在 `PROGRESS.md` 追加一行
   `| YYYY-MM-DD HH:MM | <正在做的事> | 进行中 |`（含 `nvidia-smi` 可用卡快照，FEP #8）。
2. 每完成一个 Gate 条目：`RESULTS.md` 记录数据 → `PROGRESS.md` 勾选该 Gate → `todo_write` 更新状态。
3. 遇到与蓝图不符的事实：**先更新蓝图**（升版本 + 修订记录），再改代码；禁止"代码已变、文档没变"。
4. 每个 hypothesis（H1–H4）在 `RESULTS.md` 中常驻一行"当前判定"，随数据更新。

---

## 10. 决策记录（Decision Log）

> 每次改变实验路线都必须在此留痕，编号 `D<n>`，正文相应位置引用。

| ID | 日期 | 决策 | 理由 | 影响 |
|---|---|---|---|---|
| **D1** | 2026-09-23 | 采纳**两阶段实验协议**（SDD 筛选 → PLS 确认） | 单卡 6GB 时代成本约束；v2.0 复核维持（统计逻辑不因硬件变快而改变） | §2.4、M4/M5/M6 排期；v2.0 预算升级：阶段 A 8 epoch@bs8，阶段 B 12k iter@bs8 |
| **D2** | 2026-09-23 | 使用 **COCO 迁移初始化**，且**所有 arm 统一同一份权重** | 缩短训练时长；保证跨 arm 公平 | §6.1 加载协议、A10 |
| **D3** | 2026-09-24 | **实验环境迁移到 Linux 工作站**（4×3090，conda `fastdiff`，torch 2.1.2+cu121，d2 0.6 源码编译），M0–M2 在 Linux 端重跑/补跑 | Windows 3060 6GB 是硬瓶颈；本机吞吐 0.30 s/it@bs2，eval 延迟 37.6 ms | 全文路径与环境事实；M0 baseline 本机重训 AP=61.83 为 M2+ 唯一基准 |
| **D4** | 2026-09-25 | **v2.0 训练策略升级**：bs 2→8、BASE_LR ×4=1e-4、NUM_WORKERS=4、arm-per-GPU 并行（宽度 1–2）、预算以 epoch 计、新增 E0-CTRL 控制臂、新建 `sdd.res50.bs8.yaml` | 24GB 显存利用率 <15% 太浪费；并行把 M3–M6 总 wall-clock 压到 ~8 h；E0-CTRL 吸收 bs/LR 变化使 F1/F2 增益可归因 | §2.5、§4 矩阵、M4 Gate、R14/R15 |
| **D5** | 2026-09-25 | **EMS 改用 forward-AD JVP 主实现**（`torch.func.jvp` 实测可用），有限差分降级为交叉校验模式 | torch 2.1.2 解除 v1.0 的 torch 1.10 约束；forward-AD 无截断误差且省一半 head 调用 | §3.4、M3 Gate 新增一致性条目、R4 解除 |
| **D6** | 2026-09-25 | **评测 seed 数维持 3**（Q5 关闭） | eval 在本机约 1–2 min/次，3 seed 成本可忽略；5 seed 收益边际 | §2.3 FEP #3 |

---

## 附：参考位置速查

| 需要的文件 | 位置 |
|---|---|
| 本仓库 | `/home/huachenghao/codes/fast-diffusiondet` |
| DDIM 主循环 / 冻结锚点 | `diffusiondet/detector.py:186-274` |
| 求解器包 | `diffusiondet/solvers/` |
| EDM 上游参考 | `https://github.com/NVlabs/edm`（sampler `generate.py`、预条件 `training/networks.py`、损失 `training/loss.py`） |
| DPM-Solver-v3 上游参考 | `https://github.com/thu-ml/DPM-Solver-v3`（`dpm_solver_v3.py`、`compute_EMS_scoresde.py`） |
| PFGM++ 上游参考 | `https://github.com/NVlabs/pfgmpp`（扰动核 `training/loss.py`、先验 `generate.py`） |
| PLS 数据 | `<repo>/datasets/PLS/Plantv2` |
| SDD 数据 | `<repo>/datasets/SDD/Strawberry` |
| 迁移权重 | `<repo>/pretained/diffdet_coco_res50.pth` |
| conda 环境 | `fastdiff`（`/home/huachenghao/.conda/envs/fastdiff/bin/python`） |
