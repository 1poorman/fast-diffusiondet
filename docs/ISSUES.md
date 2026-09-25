# fast-diffusiondet 实验问题排查记录（ISSUES LOG）

> 目的：按时间顺序记录 v2.0 实验全流程中遇到的代码崩溃、训练崩溃、环境问题与解决对策，
> 使读者能从零了解"每一步为什么这么做"。与 `PROGRESS.md`（做什么）、`RESULTS.md`（结论）
> 互补，本文档聚焦 **"出了什么问题、怎么发现、怎么解决、怎么预防"**。
>
> 覆盖时段：2026-09-24（M0/M1/M2）～ 2026-09-25（M3/M4）。
> 环境：Linux 工作站 4× RTX 3090，conda env `fastdiff`（Python 3.10 / torch 2.1.2+cu121 /
> detectron2 0.6 源码编译）。

---

## 1. 环境搭建阶段（2026-09-24 上午）

### 1.1 pip 从 download.pytorch.org 下载 torch 停滞
- **现象**：`pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121`
  在 2.2GB wheel 上下载 ~15 分钟后 socket 进入 CLOSE-WAIT，进度条不动。
- **根因**：境外 CDN 连接不稳定，pip 无断点续传。
- **解决**：改用清华 PyPI 镜像 `pip install torch==2.1.2 torchvision==0.16.2
  -i https://pypi.tuna.tsinghua.edu.cn/simple`。PyPI 版 torch 2.1.2 即 cu121 构建
  （CUDA 运行时通过 nvidia-* 依赖包分发），国内速度数分钟装完。
- **教训**：大 wheel 一律优先国内镜像；后台跑 pip 时用 `du -sb ~/.cache/pip` 两次采样
  判断是否真在下载。

### 1.2 conda 环境里已有可用 torch
- **发现**：本想从零装，扫描 `~/.conda/envs/*/lib/python*/site-packages/torch` 发现
  `fastdiff` 环境已装好 torch 2.1.2+cu121（此前某次 pip 实际完成）。
- **对策**：装大件前先 `conda env list` + 扫 site-packages，避免重复劳动。

### 1.3 detectron2 无 torch 2.x 预编译轮子 → 源码编译 nvcc 版本不匹配（三连坑）
- **现象 1**：`pip install -e detectron2`（构建隔离）报 "No module named torch"。
  - **原因**：build isolation 环境里没有 torch；编译 CUDA 扩展必须在目标环境内。
  - **解决**：`pip install --no-build-isolation -e ...`。
- **现象 2**：编译报 nvcc 版本不匹配错误。
  - **根因**：系统 `/usr/local/cuda` 是 **CUDA 13.0** nvcc，而 torch 是 cu121 运行时；
    torch 的 cpp_extension 默认取系统 nvcc。
  - **解决**：在 conda 环境内装 12.1 工具链并指 CUDA_HOME：
    ```bash
    conda install -n fastdiff -c "nvidia/label/cuda-12.1.1" cuda-nvcc cuda-cudart-dev cuda-cccl cuda-libraries-dev
    CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST="8.6" MAX_JOBS=8 \
      pip install --no-build-isolation -e detectron2
    ```
- **现象 3**：第一轮重编仍报 `cusparse.h: No such file`。
  - **根因**：只装了 nvcc 没装库头文件；且后台链式命令 `conda install ... && pip install`
    中 conda install 被中断，误判为已装。
  - **解决**：补装 `cuda-libraries-dev`（含 cusparse/cublas 等 dev 头），前台逐步执行。
- **教训**：① 后台链式命令（`a && b &`）中途失败不易察觉，长链路拆步跑；
  ② 头文件装到 `$CONDA_PREFIX/include`，编译前先 `ls $CUDA_HOME/include/cusparse.h` 验证。

### 1.4 numpy 2.x 与 torch 2.1 不兼容
- **现象**：导入 torch 时出现 numpy ABI 警告。
- **解决**：`pip install "numpy<2"`（落到 1.26.4）。
- **教训**：torch 2.1/2.2 时代与 numpy 2 不兼容，装环境先钉 numpy<2。

---

## 2. M1 采样器移植阶段（2026-09-24）

### 2.1 DPM-Solver-v3 仓库地址
- **现象**：`git clone LuChengTHU/dpm-solver-v3` 失败。
- **解决**：官方仓库是 **`thu-ml/DPM-Solver-v3`**。HTTPS 连 GitHub 不稳时走 SSH
  （`git@github.com:...`）成功。

### 2.2 移植 Bug 一：`time_0n` 漏掉 `index_list` 截取
- **现象**：`torch.linalg.inv(R)` 收到 2×1 矩阵报 shape error。
- **根因**：把原版 `torch.FloatTensor(index_list(time_lst, indexes)).cuda()` 改成
  device 安全写法时，**丢掉了外层的 `index_list(...)` 截取**，`time_0n` 用了全部
  历史缓存（比如 500 项）而不是最近 n+1 项 → R 矩阵维度错乱。
- **解决**：`torch.stack([t.detach().reshape(()) for t in index_list(time_lst, indexes)])`。
- **教训**：改造原版代码时逐行 diff 对照原文件；"重写一行"时要确认外层包裹函数全部保留。

### 2.3 移植 Bug 二：统计量 `(N+1,1,1,1)` 尾维与 3 维 box 广播升维
- **现象**：中间张量变成 `(B,1,P,4)`，后续 `inv` 报错。
- **根因**：原版面向 4 维图像 `(B,C,H,W)`，统计量带 `(1,1,1)` 尾维广播正好；
  我们的 box 是 3 维 `(B,P,4)`，`(N+1,1,1,1)` 广播出 4 维。
- **解决**：统计量统一 reshape 成 1 维 `(N+1,)`（scalar 版），并断言维度；
  M3 引入 per_coord `(N+1,P,4)` 时另行适配（见 §4.7）。
- **教训**：跨数据形状域移植代码，广播规则要重推一遍，不能靠原版的隐含形状约定。

### 2.4 移植 Bug 三（最隐蔽）：`dpm_v3_sample` 丢弃 `solver.sample()` 返回值
- **现象**：DPMv3 在线性高斯测试问题上 err=0.0 "完美收敛"，但各步数输出完全相同。
- **根因**：封装函数调 `solver.sample(x_T, ...)` 后没接收返回值，直接返回了 `x_T`
  （初始噪声）。之前所有 DPMv3 输出都是初始噪声的假象。
- **发现过程**：合成问题上各步数 err 全为 0 且相互相等 → "精确得可疑" → 逐行查封装。
- **解决**：`x = solver.sample(...)` 并返回 `x`。
- **教训**：**"过于完美"的结果与"崩溃"一样可疑**；新 solver 接入后先做输出随步数
  变化的 sanity check。

### 2.5 float32 下 ᾱ₀ 舍入为 1.0 → λ=+inf → 段错误（Segmentation fault）
- **现象**：NFE 测试（合成调度 w_min=1e-4）整个 Python 进程段错误，无 traceback。
- **根因链**：float32 下 `1/(1+1e-8)` 舍入为**精确 1.0** → `betas[0]=0` →
  `marginal_lambda=log(α/σ)=+inf` → 分段线性插值在 inf 上产生非法索引 →
  C++ 层崩溃（不是 Python 异常，所以没有 traceback）。
- **解决**：合成调度 w_min 取 ≥1e-3；真实 cosine 调度 ᾱ₀<1 无此问题。
- **教训**：① 无 traceback 的段错误优先怀疑 **inf/NaN 进了索引/插值**；
  ② 自定义极端参数的测试调度要检查 float32 表示边界。

### 2.6 线性高斯问题上 DPMv3-degenerated 数值精确，测不出收敛阶
- **现象**：修复 2.4 后，DPMv3 在线性高斯问题上 err 恒等于 0（指数积分器对该问题精确），
  阶数测量无意义。
- **解决**：换**非线性去噪器**——2 分量高斯混合先验（逐坐标独立，软指派后验均值），
  用高步数参考解（k=256）测相对误差阶。
- **结果**：实测阶 2.33 ≥ O(h²) ✅。
- **教训**：测收敛阶的问题必须与求解器的"精确解覆盖范围"不重合。

### 2.7 参考解步数接近调度表长度 → Vandermonde 矩阵奇异
- **现象**：steps=512（= 调度表长度 T）时 `inv(R)` 报 singular matrix。
- **根因**：`inverse_lambda` 是分段线性插值（在 N=1000 离散点上），T=512 步的
  logSNR 均匀网格往返映射后有 4 对 λ 重复 → 相邻时间步重复 → R 奇异。
- **解决**：参考解步数降到 256。实际使用 NFE≤10 ≪ N=1000，不会触发。
- **教训**：离散调度上"连续时间方法"的往返映射（λ↔t）在接近表长时分辨率不足。

### 2.8 测试桩的方向性错误：合成调度 index 方向造反
- **现象**：Euler/Heun 收敛阶测试首次失败。
- **根因**：把 w_max 放在 index 0（表 "index 0 最噪"），与真实 VP 调度
  （index 0 最干净、ᾱ 递减）方向相反，采样从"最干净端"开始整条轨迹错误。
- **解决**：`logspace(w_min, w_max, T)`，index 0 → w_min。
- **教训**：合成任何调度表前，先抄一遍真实调度的方向语义。

### 2.9 pytest 收集 tests/tmp 下的参考仓库
- **解决**：`tests/conftest.py` 写 `collect_ignore_glob = ["tmp/*"]`。

### 2.10 CRLF 行尾导致 python 内嵌替换补丁全部 miss
- **现象**：`s.replace(old, new)` 的补丁"成功"但文件没变。
- **根因**：Windows 时代遗留的测试文件是 CRLF，`old`（LF）匹配不上。
- **解决**：读写都带 `newline=''`，或按行号替换。
- **教训**：**补丁脚本必须 assert 匹配**，静默跳过最危险。

---

## 3. M2 矩阵评测阶段（2026-09-24 下午）

### 3.1 数据集目录命名与注册表不一致
- **根因**：磁盘上是 `datasets/Strawberry`、`datasets/Plantv2`，注册表期望
  `SDD/Strawberry`、`PLS/Plantv2`（Windows 机器结构）。
- **解决**：符号链接补齐（`datasets/SDD/Strawberry -> ../Strawberry`），零拷贝。
  SDD 1750/750、PLS 7916/2024 图像数与 M0 记录一致。

### 3.2 已训练 baseline checkpoint 未迁移
- **现象**：`model_final.pth`（AP=60.48）不存在，只有 COCO 迁移初始化权重
  （eval AP=26.08）。
- **对策**：本机重训 3500 iter（~22 min on 3090），得 **AP=61.83**。
  与 M0 的 60.48 差异来自 torch 版本/硬件（3090 vs 3060），属预期；
  **M2+ 所有数据以本机 ckpt 为基准，与 M0 数据不可跨表直接比较**。

### 3.3 Heun 二阶校正在 cosine 调度上灾难性发散（AP 47 → 5.6）
- **现象**：干跑矩阵时 Heun@NFE3 AP 从 47 级崩到 5.6，一阶法全部正常。
- **排查**：单图跟踪中间量发现 cosine 调度末端 w=σ/α 高达 **2e4**，DDIM 网格
  首步 |Δw|≈2e4；逐步打印 |x0_hat| 发现多次贴在 clamp 边界 ±2。
- **根因**：二阶校正在超大步长上放大 clamp 去噪器的不一致性（x0_hat 贴 clamp
  边界时两个评估点方向矛盾，校正项错误外推）；一阶法单点评估不受影响。
- **解决**：**步长守卫** `HEUN_MAX_DW=1.0`——|Δw| 超阈值区间退回一阶
  （多阶求解器 ramp-up 的标准做法），NFE 计数器如实记录实际 NFE。
- **教训**：EDM 类高阶法从图像域搬到低维 box 域时，**调度的动态范围**
  （w 跨 4 个数量级）必须先检查；二阶法的隐含假设是"相邻两点去噪器一致"。

### 3.4 A2 消融启动即 "No module named fvcore"
- **根因**：同秒并发启动两个长任务，环境解析竞态。
- **解决**：用环境内 python 的**绝对路径**启动（`~/.conda/envs/fastdiff/bin/python`）。
- **教训**：脚本化批量启动一律用绝对 python 路径。

### 3.5 checkpoint 文件名手误
- **现象**：文件"存在"却报 not found。
- **根因**：`model_0999` vs 实际 `model_0000999.pth`（少打一个 0）。
- **对策**：从 `ls`/`stat` 输出复制文件名，不要手打。

### 3.6 矩阵脚本不注册数据集
- **根因**：`register_all` 在 train_net 的 `main()` 里，不在 `setup()` 里。
- **解决**：独立脚本显式调用 `register_all(cfg=cfg)`。

### 3.7 cfg 冻结后写键报错
- **解决**：先 `cfg.defrost()` 再改，改完 `freeze()`。

### 3.8 DDIM 与 ODE solver 的返回形式不同
- **现象**：Gate 4 测试对 ddim 解包三元组失败。
- **根因**：DDIM 为保逐 bit 一致保留原返回形式（Instances 列表），ODE solver 返回
  `(img, outputs_class, outputs_coord)`。
- **解决**：测试按 solver 分支处理；文档写明两套返回契约。
- **教训**：接口"多态"要显式文档化。

### 3.9 patch 脚本按行替换误删函数体
- **根因**：行号定位 + 覆盖式写入，起点算错。
- **解决**：恢复整函数体重写；此后**结构性修改用整函数替换而非行号切片**。

### 3.10 stdout 缓冲导致"看似挂死"（假象）
- **对策**：判活看 `output/<dir>/log.txt`（detectron2 自己 flush）或 GPU 利用率，
  不要只看 nohup 重定向的 stdout。此问题在 §5.6 复发并浪费一轮排查。

---

## 4. M3 EMS 统计量标定阶段（2026-09-24 晚 ～ 09-25）

### 4.1 GT 框归一化的广播错误
- **现象**：`cxcywh / wh` 产生 NaN。
- **根因**：`wh` 是 `(2,)` 而框是 `(n,4)`，广播错位相除。
- **解决**：`wh = [W,H,W,H]` 四元组对齐坐标序。

### 4.2 betas buffer 是 float64 → Double/Float 混算崩溃
- **现象**：head 前向 `F.linear` 报 "expected scalar type Double but found Float"。
- **根因链**：`model.betas` 为 float64 → `NoiseScheduleVP.marginal_*` 返回 float64 →
  `inverse_lambda` 网格 float64 → t_input float64 进 head 时间嵌入。
- **解决**：① `NoiseScheduleVP` 内部统一 cast float32（顺带 M3 latency 优化）；
  ② EMS 脚本网格显式 `.float()`。
- **教训**：**跨模块传张量前显式钉 dtype**。

### 4.3 fwAD 不支持 head 内自定义 autograd Function
- **现象**：`torch.autograd.forward_ad` JVP 报 "jvp is not implemented"。
- **根因**：head 里有自定义 Function，PyTorch 2.1 forward-mode 未实现其 c++ 接口。
- **解决**：自动降级**中心差分**（蓝图 §M3 原方案，h=0.02 box 归一化域），
  降级只打印一次警告（节流）。
- **教训**：新 PyTorch 的"高级微分特性"遇自定义算子照样退回数值差分；降级路径要留痕。

### 4.4 统计循环缺 `no_grad` → OOM
- **根因**：差分前向没包 `torch.no_grad()`，每个网格点的激活图被 autograd 记录。
- **解决**：pass1/pass2 网格循环整体 `with torch.no_grad():`。
- **教训**：**"不需要梯度的循环"也要显式 no_grad**——差分法不是反向传播，
  但 autograd 图照样累积。

### 4.5 t_input 单位错误（嵌入域 vs 连续时间域）
- **现象**：最后网格点（t→1/N，σ→0）NaN。
- **根因**：`BoxNoisePredFn.__call__` 把 **t_input（(t−1/N)×1000，0..999）直接传给
  `marginal_alpha/std`（期望连续 t∈[1/N,1]）**→ 越界外推 α>1 → σ²<0 → NaN。
- **发现**：单点调试打印 alpha/sigma/ti 三者的值，发现 ti=999 对应 alpha 明显 >1。
- **解决**：内部换算 `t_c = t_input/1000 + 1/N`。
- **教训**：**同一脚本存在两套时间单位时，封装边界处必须换算**；docstring 写清入参域。

### 4.6 时间差分端点越界 + float32 末尾格点舍入重合
- **现象**：修复 4.5 后，f_d 仍有 3600 NaN，集中在网格尾部。
- **根因（两 bug 叠加）**：
  1. 端点单侧差分**方向写反**（j=0 是噪端 ti=999 却向 ti+1 差分，越到域外）；
  2. float32 下末尾两格点的 ti 都舍入为 0.0，按 **j 索引**判断端点不可靠。
- **解决**：按 **ti 值**判断端点（`ti >= 998.5` 噪端向内单侧、`ti <= 1.5` 净端向内单侧）。
- **教训**：① 差分方向要向"域内"单侧，画数轴确认；② float32 网格端点可能重合，
  判断条件用**值**不用**索引**。

### 4.7 per_coord 统计量撞上 scalar 专用的 `reshape(())`
- **现象**：M3 矩阵 NFE≥2 全崩（NFE=1 正常）。
- **根因**：corrector 里 `self.l[idx].reshape(())` 是 scalar 专用，per_coord
  `(P,4)` 被压成 () 后广播错。
- **解决**：`l_t.dim()==1` 才压标量，per_coord 保持原形状与 `(B,P,4)` 广播。
- **教训**：支持"两种形状"的字段访问按实际 ndim 分支，不要假定。

### 4.8 光滑性 Gate 判据在 l 过零点假阳性
- **现象**：`max |Δl|/|l|` 高达 2e5，看似失败，但曲线目测光滑。
- **根因**：l 在噪端过零，分母趋 0 的逐点比值必然爆炸。
- **解决**：改用相邻差分 `max/median` 比值（MC 噪声 ~O(10)，系统跳变 >1000），
  实测 276 ✅。
- **教训**：**过零序列不能用相对比值做平滑性判据**。

### 4.9 K=64 vs Gate 要求的 K=256
- **背景**：正式 EMS 跑时发现 train loader bs=2，32 batches 只有 K=64。
- **裁定**（用户参与）：先按 K=64 出结果，H3 边缘化再补 K=256
  （`--num-batches 128`，~4.5h 单卡）。
- **结果**：H3 在 NFE≥3 的灾难退化与 MC 噪声放大假设一致（RESULTS §5），
  K=256 重标定列为终版前置。

---

## 5. M4 EDM 训练范式阶段（2026-09-25）

### 5.1 detector.py `__init__` 中部插入方法定义（最严重的结构破坏）
- **现象**：`self.normalizer is not defined` 等连环 AttributeError；`edm_c_skip`
  等方法定义出现在 `__init__` 体内。
- **根因**：`replace_in_file` 插入 EDM 方法时锚点匹配到 init 中部相似片段，
  方法体插入后把 init 后半段（betas buffer 注册等）吞进方法作用域。
- **恢复**：`git checkout -- diffusiondet/detector.py` 回到 HEAD（工作区无未提交
  内容是关键前提），重新分小块施加 M4 改动。
- **教训**：
  1. **锚点唯一性**：replace 前确认 old_str 在文件中唯一（尤其 init 长函数内部）；
  2. 大文件多处插入时**每处一个独立小 patch**，插完 `ast.parse` + `read_lints`；
  3. 连环"属性未定义"错误，第一反应查**代码结构**（方法嵌进别的作用域），
     而不是逐个补属性。

### 5.2 `--opts` 参数风格不识别
- **根因**：detectron2 新版 Trainer 用**位置参数**传覆盖项。
- **解决**：`train_net.py --config-file X KEY VALUE`。

### 5.3 yaml `_BASE_` 相对路径
- **根因**：`_BASE_` 相对**该 yaml 文件自身**位置解析，不是 CWD。
- **解决**：非 configs/ 目录下的 smoke yaml 用绝对路径。

### 5.4 E1-HEUN 崩溃之一：预条件输出负宽高 → matcher GIoU 断言
- **现象**：iter≈36 `generalized_box_iou` 内 assert x1>=x0 崩溃。
- **根因**：EDM 预条件 `D = c_skip·x_t + c_out·F` 对每个坐标独立 clamp 到 [-2,2]，
  但 **w/h 分量可以为负** → xyxy 无序框。
- **解决**：`to_abs` 里对 cxcywh 后两维 `clamp(min=1e-4)`（训练预条件与 EDM 推理两处同步）。
- **教训**：**"逐坐标合法"≠"框合法"**；box 约束是耦合的（wh>0），clamp 要按语义做。

### 5.5 E1-HEUN 崩溃之二：λ(σ) 未封顶 → AMP 梯度爆炸 → NaN
- **现象**：修复 5.4 后 loss 高达 1100–2100（E0-CTRL 同期 ~4），iter≈230 再次
  GIoU 断言崩溃，权重已含 NaN。
- **根因**：EDM 理论权重 λ(σ)=(σ²+σd²)/(σ·σd)² 的 1/σ² 项在小 σ 时达 **1e3–1e4**；
  该权重为 MSE 设计，直接乘 L1+GIoU，AMP fp16 下梯度爆炸 → 权重 NaN →
  预条件 NaN → `NaN>NaN` 恒 False 绕过 clamp → 无序框断言。
- **解决**：`EDM_LAMBDA_MAX=50` 封顶 λ；`nan_to_num` 消毒预条件输入输出
  （NaN 梯度为 0，安全跳过该样本）。
- **结果**：loss 回落 ~650 并稳定，训练持续推进。
- **教训**：① 论文权重公式换损失函数时**必须检查量纲与动态范围**；
  ② AMP fp16 上限 6e4 让爆炸更早；③ "clamp 防不住 NaN"（NaN 比较恒 False），
  要 `nan_to_num` 在前。

### 5.6 E1 训练"挂死"两次：stdout 缓冲假象 + dataloader 死锁
- **假挂死**：iter 239 "停住"，实为 nohup stdout 块缓冲（官方 log.txt 在动、GPU 99%）。
- **真死锁**：进程 S 状态 25 分钟无日志，GPU 空转；判 dataloader worker 死锁
  （NUM_WORKERS=4）。
  → 重启用 `DATALOADER.NUM_WORKERS 0`（SDD 小数据集单进程足够，~20 分钟跑完）。
- **教训**：**判活只看 `output/<dir>/log.txt` 和 GPU 利用率**；真死锁先降 worker 数，
  小数据集直接 0。

### 5.7 孤儿训练进程抢卡（kill 只杀了父进程）
- **现象**：重启训练后进度不动，GPU0 99% 但新进程等不到卡；
  `nvidia-smi --query-compute-apps` 发现旧 PID 仍在占卡。
- **根因**：`kill -9 <父>` 只杀主进程，CUDA 子进程变孤儿继续占卡。
- **裁定**（用户拒绝进一步 kill）：不干预，两进程共享 GPU0 各自跑完
  （写同一 OUTPUT_DIR，以先落盘的 model_final.pth 为准）。
- **教训**：**杀训练用进程组**（`kill -- -PGID` / `pkill -f 唯一标识`），
  杀后用 `nvidia-smi --query-compute-apps` 验证卡已释放再重启。

### 5.8 `--output-dir` 参数不存在
- **解决**：detectron2 用位置 opts 覆盖 `OUTPUT_DIR X`。后按"同范式同权重"裁定
  E1-DPV3 与 E1-HEUN 共用 EDM 训练，不再单独训。

### 5.9 E1"卡死"的真相：loss_ce NaN 崩溃 + 非守护线程空转假象
- **现象**：E1-HEUN 训练 3 次"卡死"（进程 ALIVE、CPU 100%、GPU 无进展、日志不动），
  分别在 iter 219/239/459/398，无 traceback。
- **排查**：SIGINT/SIGUSR1 均无效（主线程阻塞在 GIL 不释放的调用）；
  带 `-u` unbuffered + 独立 stackdump 模块重启后，日志尾部终于露出
  `FloatingPointError: Loss became infinite or NaN`。
- **真因**：**全部 5 层 loss_ce = NaN，bbox/giou 有限**——λ 加权的大梯度经共享
  trunk 打爆分类头权重。之前"卡死"是主线程异常退出后非守护线程空转的假象。
- **解决**：① `EDM_LAMBDA_MAX` 50→20；② **EDM 训练禁用 AMP**（fp16 上限 6e4
  对 λ 加权梯度太紧），写进 `sdd.res50.bs8.edm.yaml`。
- **教训**：**"进程活着但什么都不干"≠ 挂死**，先看 unbuffered 日志尾部有没有
  未 flush 的异常；训练脚本一律 `-u`。

### 5.10 E1 模型 AP≈5 的根因：训练/推理 head 输入不一致
- **现象**：AMP/λ 修复后 E1 完整训完，但 edm_heun/dpm_v3 评测 AP 只有 4~6
  （E0-CTRL 同预算 67）。
- **根因**：EDM 预条件要求 head 输入是 `c_in(σ)·x_t`，推理路径
  `model_predictions_edm` 做了，但**训练前向直接喂原始 x_t**——
  c_in(σ) 在 σ∈[0.01,4] 上变化 0.25~5.2 倍，两侧输入分布完全错位。
- **修复**：训练前向在 head 调用前插入与推理完全相同的变换
  `x_in = clamp(c_in·x_t)`；c_skip 项乘的是 clamp 后的 x_t（与推理对齐）。
- **验证**：修复后同配置 loss 从 ~630 降到 ~320，训练正常推进。
- **教训**：**改训练范式时，把 train 前向和 inference 的 model_predictions 当
  一对契约逐行对齐**；差异一处就足以让模型学不到东西且无任何报错。

### 5.11 推理冒烟的数值健康信号（正面记录）
- 50-iter 半成品模型：edm_heun@NFE7 AP=0.095、dpm_v3(EDM)@NFE7 AP=0.14——
  量级合理（未训练模型），说明 **EDM 预条件推理链路数值健康**，可放心等训练收敛。

---

## 6. 流程与方法论沉淀（跨阶段）

1. **新求解器接入四步自检**（M1 建立，之后每次复用）：
   合成问题收敛阶 → NFE 计数器核对 → 真实模型有限性 → 与已知 solver 的等价性
   （如 Euler(eta=0) ≡ DDIM(eta=0)）。
2. **"过于完美"= 红旗**：err=0、各步数输出相同这类结果，先查封装是否返回了错误对象
   （§2.4 用了 2 小时才定位）。
3. **后台长任务三件套**：绝对路径 python、独立日志文件、判活看官方 log.txt +
   GPU 利用率（不看 nohup stdout）。
4. **杀进程用进程组**，杀完验证 GPU 释放（`nvidia-smi --query-compute-apps`）。
5. **补丁脚本必须 assert 匹配**；结构性修改优先整函数替换；改完 ast.parse +
   read_lints。
6. **box 域特有约束清单**（从图像域移植时逐条过）：
   wh>0（耦合约束）、xyxy 有序、clamp 逐坐标不等于框合法、两套时间单位边界换算。
7. **论文公式换损失函数**：检查权重动态范围（λ(σ) 对 L1/GIoU 必须封顶）+ AMP 精度。
8. **统计量 MC 标定**：K 值决定多步 g 系数的噪声放大上限；Gate 判据避开过零点比值。
