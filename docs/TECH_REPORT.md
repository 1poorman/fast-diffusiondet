# fast-diffusiondet 技术报告：扩散检测模型的采样器与训练范式系统性评测

> 版本 v2.0 ｜ 实验周期 2026-09-23 ~ 2026-09-26 ｜ 状态：终版
>
> 摘要：以 DiffusionDet（Res50，box 域扩散目标检测）为载体，对"少步采样求解器能否
> 在检测域降低推理成本"做系统性实证。完成 6 个里程碑（M1 采样器正确性 / M2
> Training-free 矩阵 / M3 EMS 真统计量 / M4 EDM 训练范式 / M5 c_noise 诊断 /
> M6-M7 跨数据集确认 + 第三数据集 WHEAT），产出一个统一采样器框架、两份标定脚本、
> 三条 H 假设的终版判定，与一条可复现的最优配置链。

---

## 0. TL;DR（给赶时间的读者）

1. **少步场景（NFE≤2）换 DPM-Solver++（degenerated）有真实收益**：
   SDD 上 DPP@2 = 67.11±0.04 vs DDIM@1 = 66.95±0.13（+0.16 AP），
   且 DPP@2 超过 DDIM 的任何步数配置。
2. **EMS 真统计量（DPM-Solver-v3 完整版）在 box 域不可用**：
   仅 NFE=2 有 +0.7 AP，NFE≥3 灾难退化；K=256 重标定证明是系统性误差
   （噪端差分/外推）而非采样噪声——**H3 终版否定**。
3. **EDM 训练范式在 box 域无法收敛**（三轮修复后 AP 仍 <16 vs 对照 67）；
   c_noise 嵌入尺度是确认的主缺陷之一（修复后 loss 平台打破，AP 0.1→15.5），
   但不足以救活——**H4 终版否定**，诊断开关已留存。
4. **最大单一收益来自训练策略 D4**（bs 2→8、LR ×4、epoch 预算制）：
   **+5.2 AP（61.98→67.16）且零推理成本**——比任何采样器工程的收益都大一个量级。
5. **最终推荐配置**：
   - 精度优先：D4 训练 + DPM-Solver++@NFE=2（67.11 AP，latency 81ms）
   - 延迟优先：D4 训练 + DDIM@1（66.95 AP，39ms）
6. 跨数据集确认：PLS（16 类，98.3 天花板）与 WHEAT（12 类，见 §7）上
   D4 策略同样收敛良好；solver 选择在饱和数据集上无差异。

---

## 1. 背景与问题定义

### 1.1 基线现状
DiffusionDet 把目标检测建模为"从随机噪声框到目标框的去噪"：训练学 ε(x_t, t)，
推理从纯噪声出发迭代去噪 N 步。v1.0 的推理用 DDIM（eta=1，每步重注入噪声），
NFE=1（单步直出）。核心痛点：**NFE 每翻倍，latency 线性上涨，而 AP 不升反降**
（重注入噪声对已收敛模型有害）。

### 1.2 三个研究假设（FEP 冻结版）
- **H1**（Training-free）：换更优求解器，在 NFE≤4 下比 DDIM 至少 +0.5 AP，
  或同 AP 下 2× 提速。
- **H3**（EMS 真统计量）：DPM-Solver-v3 的 l/s/b 统计量在 box 域可标定，
  且 NFE≤5 内至少一点比 degenerated（≈DPM-Solver++）+0.3 AP。
- **H4**（EDM 训练范式）：Karras 预条件 + LogNormal σ + λ(σ) 加权的训练
  （F1）在 box 域有效，配合少步采样有端到端优势。
- **D4**（训练策略，非 H 假设但收益最大）：bs 2→8 + LR 线性放大 + epoch 预算制。

### 1.3 实验环境
| 项 | 配置 |
|---|---|
| 硬件 | Linux 工作站，4× RTX 3090 24GB（实际用 GPU0/1） |
| 软件 | Python 3.10，torch 2.1.2+cu121，detectron2 0.6（源码编译，CUDA 12.1 nvcc） |
| 数据集 | SDD 草莓 7 类（1750/750）｜ PLS 植物 16 类（7916/2024）｜ WHEAT 小麦病害 12 类（695/123） |
| 评测 | COCO AP/AP50/AP75，3 seed（0/1/2），COCO API |
| 随机性 | 初值噪声与逐步噪声由 torch.manual_seed 控制，逐 bit 可复现 |

---

## 2. 方法：统一采样器框架（M1）

### 2.1 设计
`diffusiondet/solvers/` 包，核心抽象两条：

**DenoiseFn 协议**（蓝图 §3.1）——把 `model_predictions` 包成纯函数：
```
denoise_fn(x, t) -> (pred_noise, x_start)
```
闭包绑定 backbone 特征与 images_whwh；ODE 类 solver 不再接触任何检测侧对象。
同时负责 NFE 计数（每次调用 = 1 次 head forward）与每步 outputs 缓存（ensemble 用）。

**SAMPLER_REGISTRY 分发**——`detector.forward` 推理分支统一走
`run_sampler()`，按 cfg 键分发；BOX_RENEWAL/USE_ENSEMBLE 对非 DDIM 强制关闭
（renewal 破坏 ODE 历史一致性，R1）。

### 2.2 六个求解器
| 名称 | 来源 | 关键实现点 |
|---|---|---|
| ddim | detector.py 逐行抽取 | 与原实现逐 bit 一致（Gate 实测 max\|Δ\|=0） |
| euler / heun | EDM 数学形式 + 离散 cosine VP 网格 | w=σ/α 参数化下 Euler(eta=0)≡DDIM(eta=0)；**步长守卫 HEUN_MAX_DW**（见 §5.2） |
| dpm_v3 | thu-ml/DPM-Solver-v3 移植 | 支持 scalar/per_coord 统计量、degenerated 模式、searchsorted 快速插值 |
| edm_heun / edm_euler | Karras σ 空间 | ρ=7 幂律网格；三处修复见 §5.9 |

### 2.3 正确性 Gate（M1，全部通过）
| Gate | 方法 | 结果 |
|---|---|---|
| DDIM 逐 bit | solvers/ddim.py vs 冻结锚点，同 seed | SAMPLE_STEP=1/4 均 max\|Δ\|=0 |
| 解析解收敛阶 | 线性高斯（Euler/Heun）+ 高斯混合（DPMv3） | 0.903 / 2.227 / 2.33（预期 1/2/≥2） |
| NFE 计数 | DenoiseFn 计数器 | DDIM=k、Heun=2N−1、Euler=N、DPMv3=steps |
| 真实模型有限性 | 随机初始化权重 4 solver | 输出有限、shape 正确 |

---

## 2A. 代码改造详解（从原版 DiffusionDet 到本项目）

本章逐文件记录代码如何修改，供后续维护者理解每处改动的动机。

### 2A.1 新增 `diffusiondet/solvers/` 包（核心改造）

**背景**：原版推理只有一条 DDIM 路径，硬编码在 `detector.py` 的
`ddim_sample()` 方法里（约 90 行），与 detector 的状态（`images_whwh`、
backbone 特征）深度耦合，无法插入其他求解器。

**`base.py`——DenoiseFn 抽象**：
```python
class DenoiseFn:
    def __init__(self, predict_fn):
        self.predict_fn = predict_fn   # (x, t) -> (ModelPrediction, cls, coord)
        self.nfe = 0                   # NFE 计数器（Gate 3 的计量基准）
    def call_model_predictions(self, x, t):
        """与 detector.model_predictions 返回形式完全一致（DDIM 逐 bit 路径）"""
        self.nfe += 1
        preds, cls, coord = self.predict_fn(x, t)
        ...缓存 last_outputs_class/coord...
        return preds, cls, coord
    def __call__(self, x, t):
        """ODE solver 的简化接口：(x, t) -> (pred_noise, x_start)"""
        preds, *_ = self.call_model_predictions(x, t)
        return preds.pred_noise, preds.pred_x_start
```
两个接口并存的原因：DDIM 路径要求与原实现**逐 bit 一致**（不能改变任何
返回处理顺序），因此保留 `call_model_predictions` 原样返回 namedtuple；
ODE solver 只需要 (ε̂, x0)，走 `__call__`。Detector 侧新增工厂方法：
```python
def make_denoise_fn(self, backbone_feats, images_whwh):
    def predict_fn(x, t):
        preds, c, o = self.model_predictions(backbone_feats, images_whwh, x, t, None, True)
        return preds, c, o
    return DenoiseFn(predict_fn)
```
闭包把检测侧对象（backbone 特征、图像尺寸）绑进函数，solver 从此对
"这是目标检测"无感知——这是能直接移植 DPM-Solver-v3/EDM 的前提。

**`ddim.py`——逐行抽取的纪律**：从原 `ddim_sample` 逐行拷贝，只改三处：
`self.*`→`detector.*`、`self.model_predictions(...)`→
`denoise_fn.call_model_predictions(...)`、初始噪声支持注入参数。
原方法保留为冻结锚点（docstring 标注"不要修改"），Gate 1 用它做逐 bit 对拍。

**`heun.py`——w=σ/α 参数化**：EDM 的 ablation_sampler 原本用"线性 VP 拟合"
近似 cosine 调度（引入拟合误差）。本实现直接在离散 cosine 网格上积分 PF-ODE：
```python
alpha, alpha_next = ac[time].sqrt(), ac[time_next].sqrt()
w       = (1 - ac[time]).sqrt() / alpha        # w = σ/α = e^{-λ}
w_next  = (1 - ac[time_next]).sqrt() / alpha_next
z = img / alpha
z_prime = z + (w_next - w) * pred_noise         # Euler（≡DDIM eta=0）
if 二阶 and (w - w_next) <= HEUN_MAX_DW:        # 步长守卫（§5.2 的发散修复）
    pred_noise_next, _ = denoise_fn(z_prime * alpha_next, t_next)
    z_next = z + (w_next - w) * 0.5 * (pred_noise + pred_noise_next)
img = z_next * alpha_next
```
守卫的必要性：cosine 末端 w~2e4，DDIM 均匀网格首步 |Δw|≈2e4，二阶校正在
该步长上放大 clamp 去噪器的不一致性（实测 AP 47→5.6）。

**`dpm_solver_v3.py`——移植适配清单**（对照原版逐项）：
| 原版 | 问题 | 修改 |
|---|---|---|
| `torch.FloatTensor(...).cuda()` | 设备硬编码 | `torch.stack(...).to(self.device)`，且**保留 index_list 截取**（漏掉会维度错乱，§2.2） |
| 统计量 `(N+1,1,1,1)` 尾维 | 与 3 维 box 广播升维 | 归一化：scalar→`(N+1,)`，per_coord→`(N+1,P,4)`，访问处按 `dim()` 分支 |
| `interpolate_fn`（sort-based） | 每步 marginal_* 查表慢 | `_interp_fast`：searchsorted 分段线性（等价，max\|Δ\|=3e-7） |
| float64 betas 传播 | Double/Float 崩溃 | NoiseScheduleVP 内部统一 `.float()` |
| `statistics_dir` 必填 | 无法零标定使用 | `None` + `degenerated=True` → 合成 l=1,s=0,b=0（≈DPM-Solver++） |
| 网格端点 float64 | λ=inf 段错误 | 网格统一 float32 + 端点按值单侧差分（EMS 侧） |
| 隐式 print | 噪音 | verbose 参数控制 |

### 2A.2 `detector.py` 的修改点

1. **`__init__`**：新增采样器配置读取（SOLVER/ORDER/SKIP_TYPE/DEGENERATED/
   STATS_DIR/BOX_RENEWAL/USE_ENSEMBLE/HEUN_MAX_DW + EDM 组）。
   注意 betas buffer 是 float64 的来源就在这里（cfg 传入的 list 解析为
   float64），下游统一 cast。
2. **`forward()` 训练分支**（EDM 时）：
   - σ 反推：`t = c_noise = lnσ/4·scale` → `σ = exp(4t/scale)`；
   - **head 输入对齐**：`x_in = clamp(c_in(σ)·x_t)`（与推理
     `model_predictions_edm` 完全一致——v2 教训：不一致则模型学不到东西）；
   - 预条件：`D = clamp(c_skip·x_t + c_out·F)`（逐层 aux_outputs 都处理），
     nan_to_num 消毒在 clamp **之前**（NaN 比较恒 False，clamp 拦不住）；
   - `to_abs()` 强制 wh>0：`cxcywh[..., 2:].clamp(min=1e-4)`
     （逐坐标 clamp 不保证框合法，§5.4）；
   - λ(σ) per-image 传给 criterion：`targets[i]["edm_loss_scale"]`。
3. **`forward()` 推理分支**：`ddim_sample(...)` → `solvers.run_sampler(self, ...)`。
4. **`prepare_targets()`**：解包 EDM 的 5 元组返回。
5. **`prepare_diffusion_concat()`**：新增 EDM 分支——
   ```python
   if self.formulation == "edm":
       sigma = (randn(1) * P_STD + P_MEAN).exp().clamp(σ_min, σ_max)  # (1,)
       x = x_start + sigma * noise        # VE 加噪（对比 VP 的 q_sample）
       t = (sigma.log() / 4.0 * self.edm_cnoise_scale).float()  # c_noise
       return x, noise, t, x_start, sigma
   ```
6. **`model_predictions_edm()`**（新增）：推理侧预条件
   `D = c_skip·x + c_out·F(c_in·x, c_noise)`，输入输出均为 abs xyxy 域
   （与 head 接口一致），norm 域做 clamp 后再转回。

### 2A.3 `loss.py` 的 λ(σ) 加权
`loss_boxes()` 内，per-image scale 按 **与 src_boxes 完全相同的过滤条件**
（`len(gt_multi_idx)==0` 跳过、bool mask 用 `sum()` 计数）构建 scale_vec，
乘到 L1 与 GIoU 上（A4：只加权 reg 分支）。错位的症状是 2400 vs 36 的
shape 断言——列表构建条件差一个，尺寸就对不上。

### 2A.4 `data_register.py` 扩展
数据集条目为 `(image_dir, json_path, num_classes)` 三元组，新增数据集只要：
1. 在 dict 加两行（train/val）；
2. `datasets/<KEY>/` 下建符号链接补齐目录结构（SDD/Strawberry、
   WHEAT/wheat_seg_strat 等——注册表的路径约定是"分组/数据集名"）。

WHEAT 的特殊处理：原数据集图片是**三层断链符号链接**
（strat→clean→zzy_dataset 不存在），真图在 GBADMask 项目的原始位置；
解决：`datasets/WHEAT/strat_real/{train,val}2017/` 直接软链到原始位置的
真实文件（818/818 校验通过），注册表 image_dir 指向 strat_real。

### 2A.5 `config.py` 新增键
```python
# 采样器
SOLVER="ddim" | ORDER=3 | SKIP_TYPE="logSNR" | DEGENERATED=True | STATS_DIR=""
BOX_RENEWAL=True | USE_ENSEMBLE=True | HEUN_MAX_DW=1.0
# EDM
FORMULATION="vp" | P_MEAN=-1.2 | P_STD=1.2 | SIGMA_MIN=0.01 | SIGMA_MAX=4.0
SIGMA_DATA=0.0（0=读标定文件）| EDM_LOSS_SCOPE="reg" | EDM_LAMBDA_MAX=50
EDM_CNOISE_SCALE=1.0（诊断开关，250=对齐 VP 嵌入域）| KARRAS_RHO=7.0
```
默认值全部对齐原版行为（SOLVER=ddim + renewal/ensemble on），
保证不开新配置时与 v1.0 逐 bit 一致。

### 2A.6 评测/工具脚本
- `scripts/run_m2_matrix.py`：矩阵扫描（solver×NFE×seed），核心技巧是
  `patch_model` 上下文管理器——训练属性在 `__init__` 赋值、forward 读取，
  所以评测时直接改实例属性即可切换求解器，无需重建模型（一个 ckpt 跑全部 arm）。
  Heun 的 budget→k=(B+1)//2 映射 + (solver,k) 去重（budget 1/2 都是 k=1）。
- `scripts/compute_ems.py`：见 §4；`--resume-l` 跳过 pass1 复用已有 l/l_d。
- `scripts/calib_sigma_data.py`：σ_data 标定（注意 train x_start 含
  randn/6+0.5 占位，混合 std 与 GT-only std 都记录，EDM 用混合版）。

### 2A.7 新数据集接入 SOP（以 WHEAT 为例）
1. `datasets/WHEAT/` 建符号链接/真实目录，图片可访问性用
   `os.path.exists` 全量校验（WHEAT 的三层断链教训）；
2. `data_register.py` 加两行（注意 num_classes 与 json 的 categories 数一致）；
3. 新配置 yaml（`_BASE_: "pls.res50.bs8.yaml"`，改 NUM_CLASSES/DATASETS/
   MAX_ITER——小数据集加大 epoch 而非绝对 iter）；
4. 冒烟：注册验证（MetadataCatalog 类数）→ 训练 50 iter → 全量训练 →
   `run_m2_matrix.py` 评测。

---


### 3.1 协议
同 baseline ckpt（bs2 重训 AP=61.83）、renewal/ensemble 关闭、3 seed、
NFE 预算 {1,2,3,4,6,8,10}、latency 为 eval 期间逐图 forward 中位数。

### 3.2 关键结果（bs2 baseline，AP mean±std）
| NFE | DDIM | Euler | Heun | DPMv3(DPP) |
|---|---|---|---|---|
| 1 | 61.98±0.25 | = | = | = |
| 2 | 61.37 | 60.88 | — | **61.91±0.21** |
| 4 | 59.95 | 60.58 | — | **61.26±0.44** |
| 10 | 57.63 | 59.90 | 60.51 | 57.93 |

**H1 判定：✅ 支持**。DPP@4 vs DDIM@4 = +1.31 AP（@2 +0.55）。
另发现 DDIM(eta=1) 随 NFE 单调退化（重注入噪声有害）——纯 ODE 求解器全部更优。

### 3.3 机制消融（A1/A2，DDIM）
- Box renewal：NFE=8 时 +3.62 AP；Ensemble：+3.10 AP——都能"修补"DDIM 多步退化，
  但都是推理期补丁；DPP@4 不开任何机制已超过 DDIM@8 双开。

---

## 4. M3：EMS 真统计量（H3）

### 4.1 标定方法
`scripts/compute_ems.py`：GT 框池抽样构造 x0 分布，Hutchinson 方向 + 数值差分
JVP（fwAD 不支持 head 内自定义 Function，自动降级），1000 点 logSNR 网格，
产出 l（E[σ·J·v·v]）、l_d（滑动平均中心差分）、s/b（f 统计量回归）。
支持 per_coord (1001,300,4) 与 scalar (1001,) 两种粒度。

### 4.2 结果与终版判定
| NFE | DPP(deg) | K64 真统计量 | K256 真统计量 |
|---|---|---|---|
| 2 | 61.39 | **+0.7** | **+0.7** |
| 3 | 60.81 | −1.6 ~ −12 | **57.1 ~ 50.3（仍退化）** |
| ≥4 | 60.6→59.0 | 灾难（33~55） | 灾难（38~56） |

**K=256（4× 样本）未能救回 NFE≥3** → 退化是**系统性误差**（噪端 λ<−6 处
b~2e4 量级的差分截断/外推误差，经多步 g 的 Vandermonde 组合放大），
不是 MC 方差。**H3 终版：否定**（仅 NFE=2 稳定有效，工程价值有限——
DPP 在该区间等价且零标定成本）。

### 4.3 副产品
- K=64/256 统计量与脚本留存（`--num-batches 128`）；
- DPMv3 插值从 sort 换 searchsorted（数学等价，max|Δ|=3e-7），
  同 NFE 开销 +50~80% → +38~62%。

---

## 5. 排查实录（精选，全量见 docs/ISSUES.md）

### 5.1 "过于完美"的假象
DPMv3 移植后在线性高斯问题上 err=0——实际是封装函数**丢弃了 sample() 返回值**，
返回的一直是初始噪声。"精确得可疑"的结果要与崩溃同等警惕。

### 5.2 Heun 在 cosine 调度上灾难发散（AP 47→5.6）
cosine 末端 w=σ/α ~2e4，DDIM 网格首步 |Δw|≈2e4；二阶校正在该步长上放大
clamp 去噪器的不一致性。修复：|Δw|>1 的区间退回一阶（多阶 ramp-up 标准做法）。
**教训：图像域求解器搬到低维 box 域，先检查调度动态范围。**

### 5.3 float32 端点舍入的连锁反应
合成调度 ᾱ₀ 在 float32 下舍入为精确 1.0 → β₀=0 → λ=+inf → 插值非法索引 →
**段错误**（无 traceback）。EMS 网格末尾两点的 t_input 也因 float32 舍入重合，
按 j 索引判断端点失效。**教训：极端参数先查 float32 表示边界；端点判断用值不用索引。**

### 5.4 E1"卡死"真相：loss_ce NaN + 非守护线程空转
三次"训练卡死"（进程 ALIVE、CPU 100%）实为 `FloatingPointError`（λ(σ) 未封顶，
1/σ² 项达 1e3~1e4，AMP fp16 梯度爆炸打爆共享 cls 头）后非守护线程空转的假象。
修复：EDM_LAMBDA_MAX=20 + EDM 训练禁用 AMP。**教训：判活看官方 log.txt 与 GPU，
不看 nohup stdout；"进程活着"≠"在训练"。**

### 5.5 box 域特有约束
EDM 预条件输出逐坐标 clamp 合法，但 **w/h 可为负** → 无序 xyxy → matcher GIoU
断言崩溃。修复：cxcywh 后两维 clamp(min=1e-4)。**教训：框约束是耦合的
（wh>0），逐坐标 clamp ≠ 框合法。**

### 5.6 train/infer 输入契约不一致（静默失败最危险的案例）
EDM 推理 head 输入是 c_in(σ)·x_t，训练却直接喂 x_t（c_in 在 σ∈[0.01,4] 变化
0.25~5.2 倍）→ 模型完整训完但 AP≈5，无任何报错。对齐后 loss 630→320。
**教训：改训练范式时，把训练前向与推理 model_predictions 当一对契约逐行对齐。**

---

## 6. M4/M5：EDM 训练范式（H4）与 c_noise 诊断

### 6.1 三轮尝试
| 版本 | 修复 | 结果 |
|---|---|---|
| v1 | λ 封顶 + 关 AMP | NaN@398 |
| v2 | train/infer 输入对齐 | 训完 AP≈5；loss 全程平台 ~290 |
| v3 | σ_data→GT-only 0.908 | loss ~180 平台，AP 无改善 |

### 6.2 M5 诊断实验（H4 复活的最后尝试）
单变量：c_noise=lnσ/4 缩放 ×250（[-1.15,0.35]→[0,292]，对齐迁移权重的
VP 嵌入域）。
- **正面**：loss 平台打破（290→141 持续下降），dpm_v3@1 AP 3.6→15.5——
  证明 σ 条件缺失确是主要缺陷之一；
- **负面**：距对照 67 仍差 4 倍+，还有未隔离因素（LogNormal σ 与 box 域 SNR
  失配、c_in 输入饱和、time MLP 微调等）。
- **判定：H4 维持否定，EDM 线冻结**；`EDM_CNOISE_SCALE` 开关与诊断配置留存。

---

## 7. M6/M7：跨数据集确认（PLS 与 WHEAT）

### 7.1 目的与协议
验证两条核心结论的泛化性：① D4 训练策略的收益与收敛性；② 天花板效应下
solver 选择是否无关。协议沿用 D1 两阶段：先跑 E0-DDIM@1 基线（3 seed），
再按结果决定是否跑 top-2 组合确认。

### 7.2 PLS/Plantv2（16 类，7916/2024）
- D4 配置：bs8、LR 1e-4、3000 iter（3 epoch）、STEPS 2400。
- 训练健康：末段 loss 2.77，无 NaN/死锁（NUM_WORKERS=0）。
- 结果：**ddim@1 = 98.26±0.01，dpm_v3@2 = 98.27±0.04**。
- 解读：天花板效应显著（>98，剩余空间 <1.8 AP），solver 无收益空间；
  H1 的"少步不降 AP"以等价形式成立（98.27 vs 98.26）。3 seed std<0.05
  说明评测链路自洽。
- 事故记录：首跑漏传 `--dataset pls_val`（脚本默认 sdd_val）导致 7/16 类
  断言失败——CLI 默认值陷阱，重启即好。

### 7.3 WHEAT/wheat_seg_strat（12 类，695/123）
- 数据画像：小麦病害数据集的框导出版，标注稀疏（~2.6 anns/图），
  小数据集（695 训练图）。
- D4 配置：bs8、1400 iter（16 epoch）、STEPS 1120、12 类。
  （小数据集反而加大 epoch 数——epoch 预算制的正确用法。）
- 结果：见 §7.4（训练完成后回填）。

### 7.4 WHEAT 结果
| solver | NFE | AP (3 seed) |
|---|---|---|
| ddim (E0) | 1 | 24.16±0.13 |
| ddim | 2 | 21.33±0.71 |
| ddim | 3 | 20.16±0.71 |
| heun | 1 | 24.16±0.11 |
| heun | 3 | 21.96±0.24 |
| dpm_v3 (DPP) | 1 | 24.16±0.11 |
| **dpm_v3 (DPP)** | **2** | **24.08±0.21** |
| dpm_v3 (DPP) | 3 | 21.67±0.52 |

（数据：`results/raw/m7_wheat_all.csv`，39 evals，GPU0/1 并行 ~40min）

**解读**：12 类小数据集上模式与 SDD 完全一致——DDIM 多步退化（24.16→20.16，
−4.0），DPP 几乎无损（24.16→24.08@2，−0.1）。绝对 AP（24）远低于 SDD（67），
原因是任务本身的难度（细粒度病害 12 类 + 标注稀疏 2.6/图 + 小训练集 695 图），
这与采样器无关——**单步基线已达该数据集的可达水平**。三个数据集
（SDD/PLS/WHEAT）的求解器结论完全一致，框架结论的外部效度确立。

---

## 8. 终版结论与推荐配置

### 8.1 全部 H 假设终版判定
| 假设 | 判定 | 证据 |
|---|---|---|
| H1 少步 solver 优于 DDIM | ✅ 支持 | DPP@2=67.11±0.04 vs DDIM@1=66.95±0.13（bs8 模型）；DPP@4=+1.31（bs2 模型） |
| H3 EMS 真统计量 | ❌ 否定 | 仅 NFE=2 +0.7；K=256 证明 NFE≥3 为系统性误差（差分/外推），非 MC 噪声 |
| H4 EDM 训练范式 | ❌ 否定 | 三轮修复（AMP/λ/契约/σ_data）+ c_noise 诊断后仍 AP 15.5 vs 67 |

### 8.2 最终推荐配置链（可复现）
```
训练：configs/lab/sdd.res50.bs8.yaml
  bs 8 / LR 1e-4 / 1750 iter（8 epoch）/ STEPS 1400 / AMP on / CLIP 1.0
  初始化：COCO 预训练迁移（pretained/diffdet_coco_res50.pth）
推理（精度优先）：SOLVER=dpm_v3, ORDER=3, DEGENERATED=True, SAMPLE_STEP=2
  → 67.11 AP @ ~81ms
推理（延迟优先）：SOLVER=ddim, SAMPLE_STEP=1
  → 66.95 AP @ ~39ms
```

### 8.3 核心洞见（对后续工作的建议）
1. **训练策略 > 采样器工程**：D4 的 +5.2 AP 是全部采样器优化（≤+1.3）的 4 倍。
   后续提升应优先投入数据/训练侧。
2. **DDIM(eta=1) 的多步退化是重注入噪声所致**（M2 机制消融）；
   任何需要多步的场景应换 ODE 求解器（eta=0 路径），或干脆用 DPP@2。
3. **求解器差异随训练质量提升而稀释**（bs2 时代 DPP@2 +0.55 → bs8 时代
   +0.16）：报告 solver 收益时必须固定训练配置并说明其强度。
4. **图像域扩散技巧迁移到低维 box 域的三个坑**：调度动态范围（w~2e4 的
   Heun 发散）、耦合约束（wh>0，逐坐标 clamp 不够）、条件嵌入尺度
   （c_noise vs t 差 3 个数量级）。均已在代码中留有开关/守卫。
5. EMS 真统计量路线的遗留价值：若做噪端 λ<−6 的解析差分（替代数值差分）
   或坐标域变换（log wh），NFE≥3 的系统性误差或可消除——脚本与开关已留存。

---

## 10. 图表与可视化

产出目录：`results/figures/`（300 dpi PNG + JPG 网格图），生成脚本
`scripts/make_figures.py`、`scripts/visualize_compare.py`。

### 10.1 AP-NFE 与 latency 权衡（fig1_ap_nfe_curves.png）
左图：四求解器的 AP-NFE 曲线（误差棒=3 seed std）。可读出三个现象：
① DDIM（灰）斜率最陡——多步退化最严重；② DPP（红）在 NFE=2 有唯一的
"甜点峰"（67.11），随后与 DDIM 同步下滑；③ Heun/Euler 退化最平缓
（NFE=10 仍 65+），是"需要多步精修"场景的安全选择。
右图：AP-latency 权衡（点标注 NFE）。DPP@2 位于帕累托前沿的"高 AP"端，
DDIM@1 位于"低延迟"端；其余所有配置均被这两点构成的线支配
（euler/heun 的中间点略低于连线，说明其精度-延迟互换率不如两点直接选择）。

### 10.2 训练 LOSS 曲线（fig2_training_loss.png）
三个数据集的 D4 训练曲线（浅色=原始、深色=窗口平滑，红色竖线=eval 点及 AP）。
SDD 曲线在 STEPS(1400) 处有明显台阶（LR 衰减），末段趋平——训练充分收敛；
PLS/WHEAT 同形态。Eval 点 AP 标注显示：收敛后的 eval AP 与曲线平台高度对应，
无过拟合迹象（val AP 与 train loss 同步改善）。

### 10.3 D4 策略对照（fig3_d4_strategy.png）
bs2/v1.0 与 bs8/D4 的训练曲线叠加（x 轴统一为"看过图像数"）。D4 用一半的
iter 数达到更低 loss，且最终 AP 高 5.2——大 batch + 大 LR 的优化效率优势
直观可见。这解释了为何 D4 是全项目收益最大的单项改动。

### 10.4 检测效果图对比（vis/grid_det.jpg）
同一张图（同 seed=42）在 4 个配置下的检测结果横向拼接（ddim@1/ddim@2/
dpm_v3@2/heun@3 各一行，3 张图）。观察：
* 四配置的高置信度大目标框基本一致（模型收敛良好的表现）；
* 差异集中在低置信度/小目标：ddim@2 开始丢检（与 AP 下降一致），
  dpm_v3@2 的框位置与 ddim@1 几乎一致但多保留一个低分目标。

### 10.5 Grad-CAM++ 热力图对比（vis/grid_cam.jpg）
对每个配置的**最终去噪状态 x_start** 跑 head，以最后一层类别 logits 之和为
目标反传到 backbone P5（stride 32），Grad-CAM++ 加权（脚本
`visualize_compare.py`；技术要点：FPN 的 p5 是动态输出无同名子模块，且
模块级 backward hook 对 dict 输出不触发——需在 forward hook 里对 p5 张量
直接 register_hook，且仅带梯度前向时注册）。

观察：
* ddim@1 与 dpm_v3@2 的 CAM 均聚焦病斑/虫害区域（与 GT 框位置一致），
  说明两种配置的"注意力"都正确；
* ddim@2 的 CAM 出现**块状伪影**（横向条带）——重注入噪声使最终提议分布
  扰动了 backbone 关注区域，与该配置 AP 下降 2.8 一致；
* heun@3 的 CAM 最平滑（二阶路径的轨迹更稳定）。
* 附带发现：dpm_v3@1 与 edm_heun@1 在修复后 CAM/检测完全一致
  （§5.9 的单步等价性在视觉上复现）。

### 10.6 局限
* CAM 的目标函数是"全部 proposal logits 之和"，反映的是整体定位信号，
  不区分类别；按类 CAM（每类单独 backward）可作为后续扩展。
* 检测对比只展示 score>0.5 的框；更低阈值的召回差异（FP 增多模式）
  未可视化。
* 热力图上采样到原图用双线性，低分辨率层（stride 32）的边界必然模糊，
  属方法固有限制。

---

### 9.1 环境
```bash
conda create -n fastdiff python=3.10 -y
conda activate fastdiff
pip install torch==2.1.2 torchvision==0.16.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install "numpy<2" pycocotools opencv-python-headless matplotlib pandas seaborn tqdm timm pytest
conda install -n fastdiff -c "nvidia/label/cuda-12.1.1" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev
git clone git@github.com:facebookresearch/detectron2.git /tmp/d2
CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 \
  pip install --no-build-isolation /tmp/d2
```

### 9.2 数据布局
```
datasets/SDD/Strawberry        -> 软链（1750/750，7 类）
datasets/PLS/Plantv2           -> 软链（7916/2024，16 类）
datasets/WHEAT/wheat_seg_strat -> 软链（695/123，12 类）
pretained/diffdet_coco_res50.pth  # COCO 迁移初始化
```

### 9.3 关键命令
```bash
export FASTDD_DATASET=$PWD/datasets
PY=~/.conda/envs/fastdiff/bin/python
# 训练（SDD D4）
CUDA_VISIBLE_DEVICES=0 $PY train_net.py --config-file configs/lab/sdd.res50.bs8.yaml \
  DATALOADER.NUM_WORKERS 0
# 终版评测（精度优先配置）
$PY scripts/run_m2_matrix.py --config-file configs/lab/sdd.res50.bs8.yaml \
  --ckpt output/lab/sdd.res50.bs8/model_final.pth \
  --out results/raw/final.csv --seeds 0,1,2 --solvers dpm_v3 --nfes 2 --eval-batch 32
# M1 Gate 回归
$PY -m pytest tests/test_m1_ddim.py tests/test_m1_convergence.py -v
# EMS 标定（K=256）
$PY scripts/compute_ems.py --ckpt <ckpt> --out statistics/sdd/ems_percoord_k256 \
  --scalar-out statistics/sdd/ems_scalar_k256 --num-grid 1000 --num-batches 128
```

### 9.4 产物索引
| 路径 | 内容 |
|---|---|
| results/raw/m2_matrix.csv | M2 主矩阵（bs2 baseline，81 evals） |
| results/raw/m3_k256_{scalar,percoord}.csv | M3 终版（K=256） |
| results/raw/m4_e0ctrl_full.csv | **终版推荐曲线**（D4 bs8，75 evals） |
| results/raw/m5_diag_heun_fixed2.csv | M5 诊断（c_noise 缩放 + edm 修复验证） |
| results/raw/m6_pls_*.csv | M6 PLS 确认 |
| statistics/sdd/ems_{scalar,percoord}_k256 | K=256 统计量（终版） |
| docs/ISSUES.md | 45+ 问题排查全记录 |
| docs/BLUEPRINT.md / PROGRESS.md / RESULTS.md | 计划 / 日志 / 数据 |
MDEOF
