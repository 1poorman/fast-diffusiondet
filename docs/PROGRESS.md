# 工作状态实时记录

> 配套文档：`BLUEPRINT.md`（蓝图/方案/验收标准）、`RESULTS.md`（实验数据，M2 起）
> 更新规则：**每次动手前先加一行"进行中"，完成后改状态并勾 Gate。数字变化先落 `RESULTS.md`。**

最后更新：**2026-09-24 08:56** ｜ 当前里程碑：**M0 已完成 → 待进入 M1**

---

## 1. 里程碑总览

| 里程碑 | 状态 | 目标摘要 | 关键 Gate |
|---|---|---|---|
| **M0** 环境与基线建立 | ✅ **完成**（2026-09-24） | 接入 PLS/SDD、清理旧 bug、跑通单卡训练 | ✅ 全部达成：SDD baseline **AP=60.48**；it/s、latency、显存已入库 |
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

## 3. 排期估算（基于 M0 实测）

**实测基准**：训练 ≈ **2.0 it/s**（0.50 s/it，bs=2）；单次 val 评测（SDD 750 张）≈ **1.5 min**；
训练显存峰值 **3.33 GB**（余量充足，可试 bs=4 把训练再提速约 40%）。

| 里程碑 | 预估 GPU 时长 | 计算依据 |
|---|---|---|
| M0 | ✅ 已完成（≈2 h 含调试） | — |
| M1 | ~0（无 GPU） | 采样器重构 + 解析解单测，纯 CPU |
| M2 | **≈ 2.5 h** | 4 solver × 7 NFE × 3 seed = 84 次 eval × 1.5 min，另加 latency bench |
| M3 | **≈ 3 h** | EMS 计算（1000 网格 × K 样本，目标 ≤2 h）+ 重跑 eval |
| M4 | **≈ 2 h** | EDM 主 arm + A4/A5/A6 各一个 arm ≈ 4 × 30 min |
| M5 | **≈ 3.5 h** | D ∈ {32,128,512,2048,∞} 五个 arm + A7 两个 ≈ 7 × 30 min |
| M6 | **≈ 5 h** | PLS 12000 iter × 3 arm（baseline + top-2），≈1.7 h/arm |
| **合计** | **≈ 16 h GPU**（不含调试往返） | |

> ⚠ 上表未含调试往返与失败重跑，实际按 **1.5×** 估 → 约 **24 h GPU**。
> 若要压缩：① bs 提到 4（训练提速 ~40%）；② M6 的 PLS `MAX_ITER` 从 12000 降到 8000；
> ③ M2 的 seed 从 3 降到 1（省 2/3）。这三项已在 `PROGRESS.md` §7 Q5 与蓝图 §5 记录为可调项。

---

## 4. Gate 检查清单

### M0 Gate
- [x] `eval-only` 能跑出非空 AP → **27.91**（未训练，仅迁移初始化）
- [x] 训练 100 iter 无 OOM，loss 正常下降 → total_loss 15.85 → 8.1，max_mem 2.0 GB
- [x] SDD 短周期 baseline checkpoint 产出 → **AP = 60.48 @ 3500 iter**
- [x] `it/s` / `latency_ms(NFE=1)` / `peak_mem_GB` 三项已入 `RESULTS.md` → 2.0 it/s / 108.48 ms / 3.33 GB
- [x] 排期估算已填表

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
| **H0** | head forward 是延迟主项，NFE 与延迟线性相关 | ✅ **支持** | 边际 71.5 ms/step，占 NFE=1 延迟 66%（RESULTS §2） |
| H1 | NFE≤4 时 2 阶+ 求解器 AP ≥ DDIM +0.5 | ⬜ 待测（M2） | — |
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
| — | 建立 `fast-diffusiondet/` 并 fork 源码（排除 `petrain/`、`output/`，共 43 文件 8.41 MB） | ✅ |
| — | 校验 SSH（`git@github.com` 认证成功，账号 `1poorman`）；远端确认为**空仓库** | ✅ |
| — | 归档不可运行的上游实验代码 → `legacy/experimental/`，并解除 `__init__.py` 依赖 | ✅ |
| — | 写 `.gitignore` / `.gitattributes` / `README.md`(fork 说明) | ✅ |
| — | git 三提交：`376e325` 基线 → `ec19193` 文档 → `de568bd` 基建与隔离 | ✅ |
| — | 关联远端并推送 `main`；建立 `dev` 分支 | ✅ |
| — | 改写全部历史 author/committer → `1poorman <2990161445@qq.com>`（`git rebase --root --exec`，4 commits） | ✅ |
| — | **决策 D1**：采纳两阶段协议（SDD 筛选 → PLS 确认） | ✅ 已入蓝图 §2.4 / §10 |
| — | **决策 D2**：采用 COCO 迁移初始化，且所有 arm 统一同一份权重 | ✅ 已入蓝图 §6.1 / §10 |
| — | 补充 §6.1 预训练权重加载协议（哪些层匹配/哪些必须丢弃）+ 新增 A10/A11 消融 | ✅ |

### 2026-09-24（M0 实施）

| 时间 | 事项 | 状态 |
|---|---|---|
| 00:13 | 100 iter 训练冒烟：前向/反向通过，max_mem 2.0 GB | ✅ |
| 00:14 | 修 `CHECKPOINT_PERIOD=0` 导致的 `ZeroDivisionError` | ✅ |
| 00:20 | **eval-only 跑通**：未训练即 AP=27.91 / AP50=81.78 | ✅ |
| 00:33 | 提交 `34450c0`：数据集注册 + 迁移初始化 + lab 配置 | ✅ |
| 00:39 | 测吞吐：200 iter / 172 s → **0.425 s/it**，显存 3.32 GB | ✅ |
| 08:25 | 训练 0→1000 iter，**AP=45.94** | ✅ |
| 08:36 | 训练 →2000 iter，**AP=50.75** | ✅ |
| 08:46 | 训练 →3000 iter，**AP=59.49**（LR 衰减后跳涨 +8.7） | ✅ |
| 08:54 | 训练 →3500 iter 完成，**最终 AP=60.48 / AP50=82.21** | ✅ |
| 08:56 | 延迟基准：NFE=1/2/4 → **108.5 / 185.8 / 323.0 ms**，显存 0.71 GB | ✅ |
| 08:56 | 产出 `docs/RESULTS.md` + `results/raw/*.csv` | ✅ |

**本轮新发现**
7. **head forward 是延迟主项**：边际每步 ≈ **71.5 ms**，占 NFE=1 总延迟的 **66%**
   （backbone+其他仅 ~37 ms）。→ 新增假设 **H0 已判定为支持**，项目立论成立。
8. **Windows + detectron2 的两个坑**（均已固化到代码注释与 README）：
   ① fvcore 用 locale 编码（GBK）读 yaml → **`configs/` 下必须全 ASCII**；
   ② `COCOEvaluator` 直接从 json 重建类别 → 评测 split **必须**用剔除 background 的干净 json。
9. 显存余量充足（训练 3.33 GB / 6 GB），**bs 可提到 4** 换取 ~40% 训练提速。
10. LR 衰减点（2800 iter）对短周期训练影响极大（+8.7 AP），
    → FEP §2.3 要求所有 arm **必须共用同一 `STEPS`**，否则对比失效。

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
6. **修正蓝图 B4**：`samplers/l.npz`(2 MB)、`sb.npz`(5.5 MB) **确实存在**，但实测形状为
   **`(121, 3, 32, 32)`**，是 **CIFAR-10 图像的 EMS 统计量**，与 box 数据 `(B,P,4)` 完全不兼容。
   → 不能走"直接拿来用"的捷径，M3 必须自算；这两个文件已随 `legacy/` 归档并被 `.gitignore` 排除。

**M0 已全部完成 ✅。下一步：进入 M1（采样器统一接口与数值正确性）**
1. 建 `diffusiondet/solvers/`：`base.py`(denoise_fn 协议 + NFE 计数) / `schedule.py` / `ddim.py` / `heun.py` / `dpm_solver_v3.py`
2. **DDIM 逐 bit 等价回归**：新 `ddim.py` 与 `detector.py:186-274` 同 seed 输出 `max|Δ| < 1e-5`
3. 解析解回归单测：用已知闭式解的去噪器验证 Euler O(h) / Heun O(h²) / DPMv3 ≥ O(h²)
4. `heun.py` 走 `edm-main/generate.py:66-176` 的通用形式，钉 `discretization='vp', schedule='vp', scaling='vp'`
5. 新增配置项 `SOLVER / ORDER / SKIP_TYPE / DEGENERATED / STATS_DIR`

> baseline 已就位：`output/lab/sdd.res50/model_final.pth`（SDD **AP=60.48**，NFE=1，108.5 ms）。
> M2 的 training-free 对比矩阵可直接复用它。

> 注：原 `train.py` 已归档至 `legacy/`（硬编码 WGISD 路径），
> 一律使用官方入口根目录 **`train_net.py`**，数据集注册改为包内模块 + CLI 参数。

---

## 7. 阻塞与待决问题

| # | 问题 | 状态 | 影响 |
|---|---|---|---|
| Q2 | ~~是否隔离 `detector_dpm3.py`/`detector_noise.py`/`dmp3.py`/`train.py`？~~ | ✅ **已办**（2026-09-23） | 已归档 `legacy/experimental/` 并解除 `__init__.py` 导入 |
| Q3 | ~~是否允许修改原 `configs/*.yaml`？~~ | ✅ **已定**（2026-09-23） | 一律新建到 `configs/lab/`，原 configs 保持 upstream 原样 |
| ~~Q1~~ | **两阶段协议** | ✅ **已采纳**（2026-09-23）→ **D1** | SDD 跑全矩阵筛选，PLS 只跑 top-2 + baseline |
| ~~Q4~~ | **是否用 `diffdet_coco_res50.pth` 迁移初始化** | ✅ **采用**（2026-09-23）→ **D2** | 所有 arm 统一同一份权重；新增 A10 对照从头训练 |
| Q5 | 评测 seed 数量固定为 3 是否可接受？（影响 eval 总时长约 3×） | 待用户确认 | M2/M3 排期 |

> 编号说明：用户答复中的 "Q2 迁移初始化" 对应本表原 **Q4**（因 Q2/Q3 已在上一步结办）。

---

## 7.1 版本管理约定

| 项 | 值 |
|---|---|
| 远端 | `git@github.com:1poorman/fast-diffusiondet.git` |
| 本地路径 | `e:/Files/DL-code/model+/diffusion-model/fast-diffusiondet` |
| 默认分支 | `main`（只接受已过 Gate 的成果） |
| 日常分支 | `dev`；按里程碑开 `feat/m0-*`、`feat/m1-*`、`feat/m2-*` …，过 Gate 后合入 `dev` |
| 发布 | `dev` 达成一个里程碑 → PR 合入 `main` → 打 tag `m<n>` |
| **Commit 规范** | `<type>(<milestone>): <subject>`，type ∈ `feat` `fix` `perf` `docs` `chore` `test` `refactor` |
| 提交身份 | ✅ **已确认为** `1poorman <2990161445@qq.com>`，全部历史已改写（2026-09-23）<br>⚠ 本机未设 `git config user.*`，后续提交需带 `-c user.name=1poorman -c user.email=2990161445@qq.com`（或自行 `git config --local user.email 2990161445@qq.com`） |
| 不在版本库 | `petrain/`、`output/`、`statistics/`、`*.pth`、`*.npz`（见 `.gitignore`） |

**分叉基线：`376e325`** —— 该 commit 是 upstream 源码快照，
任何"我们改了什么"都可以 `git diff 376e325` 一眼看清。

**每次动手的流程（写死，照做）**
```
1. git checkout dev && git pull
2. git checkout -b feat/m<n>-<what>
3. 改代码 → PROGRESS.md 加日志行 → RESULTS.md 落数据
4. git add -A && git commit -m "<type>(m<n>): ..."
5. 过 Gate → 合入 dev → 里程碑完成时 PR 到 main 并 tag
```

---

## 8. 结论落袋区（随 H1–H4 判定推进，M6 汇总）

> 暂空。每个 hypothesis 判定完成后在此写一句话结论 + 指向 `RESULTS.md` 的数据表编号。
