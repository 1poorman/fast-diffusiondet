# G2 初筛：真实四级学生的质量筛选蒸馏

日期：2026-10-11。承接 G1 的 H4 精度缺口，用户“继续”授权后实现并启动。
结果统一写 [RESULTS.md](RESULTS.md)，本文定义实际运行协议。

## 模型与初始化

- 学生是 `NUM_HEADS=4` 的真实四级结构，保留其全部三级辅助检测损失；不是六级模型的评测早退。
- 教师为独立的六级模型，来自现存 SDD bs8 短训 ckpt，`eval()`、`requires_grad_(False)`、`no_grad()`。
- 学生只删除源权重 `head.head_series.4.*`、`head.head_series.5.*` 的 84 个 tensor；
  其余参数严格加载。缺分类头、形状不匹配、非预期多余权重均报错。
- 教师作为显式 `forward(..., teacher=teacher)` 参数传递，不注册在学生模块中、不写入学生检查点。
- 学生和教师使用同一增强图像、完全相同的训练噪声框和 t；教师独立计算其冻结 backbone 特征。
  不把学生不断变化的特征作为教师输入。两者均维持原 uniform 训练 t，尚未叠加 t 对齐。
- 本轮是从同一训练过的权重继续微调；不同训练 seed 改变数据顺序、增强和噪声，
  **不代表从随机初始化独立训练的种子方差**。

## 匹配与损失

教师与学生分别使用原 Dynamic-K 匹配。对每个 GT，在分配给该 GT 的教师正 query 中
选择 IoU 最大的一个，再与匹配到同一 GT 的学生正 query 配对。proposal 序号可以不同。

只有同时满足以下条件的配对参与 KD：

1. 教师有匹配到该 GT 的正 query；
2. 教师 IoU ≥ 0.5，且严格高于该学生 query 的 IoU；
3. 教师最高分预测类别等于该 GT 类别。

`L_KD_cls = T² * sum_classes KL(Bernoulli(sigmoid(z_t/T)) || Bernoulli(sigmoid(z_s/T)))`

`L_KD_box = 5 * L1(normalized_xyxy_s, normalized_xyxy_t) + 2 * (1 - GIoU(s,t))`

`L_total = 原有加权检测损失（含各级辅助损失） + 0.5 * L_KD_cls + 1.0 * L_KD_box`

T=2，KD 按 batch 内合格正配对数归一化，背景不参与。
教师输出、质量阈值和匹配均 stop-gradient；空 GT 或无合格配对返回可求导零值。
本原型只在学生最后一级施加 KD，不自称复现了 D-FINE 的分布 GO-LSD。
训练脚本限定单 GPU，当前正样本归一化未实现 DDP 跨 rank 聚合。

## 配对实验

| 项 | 协议 |
|---|---|
| 臂 | `control`：只用原检测损失；`kd`：增加上述质量筛选 KD |
| 训练 seed | 10、11、12，分别配对 |
| 预算 | 每臂 200 iter × batch 8 = 1600 图次，约 0.91 epoch；短预算初筛 |
| 优化器 | 新建 AdamW，LR=2.5e-5，weight decay=1e-4，梯度范数裁剪=1 |
| LR | 前 20 步线性 warmup（0.1→1），第 160 步起乘 0.1 |
| 数值 | FP32，TF32 off；训练 cuDNN deterministic on；评测恢复 G1 的默认算法模式 |
| 数据 | 同一 SDD train；原随机翻转和多尺度 resize；不改 sampler/类重采样策略 |
| 配对保障 | 每步重置 Python/NumPy/torch 种子；验证实际增强图像、GT、image_id 的 SHA256 |
| 教师检查 | 起止 state digest 一致，全部梯度为 None |
| 评测 | 每个固定末步 checkpoint：750 图 × 推理 seed 0/1/2；与 G1 同 image_id 噪声协议 |
| 计时 | batch=1，20 warmup + 同 200 图；只包含模型调用，与 G1 边界一致 |

200 步初筛检验方向与可训练性，**不等于蓝图中完整预算训练的 G2 Gate**。
KD 收益先按每个训练 seed 内的 3 个推理 seed 求平均，再计算 3 个配对差值；
不能把 9 次推理结果当作 9 次独立训练。
输入和初始化有实际 hash 校验；cuDNN 确定性设置不保证 ROIAlign 等全部 CUDA
反向算子逐 bit 复现，因此不能把本协议理解为消除了所有训练运行噪声。
按预先固定的末步评测，不根据验证集挑选中间最优 checkpoint。

## 实现和验证

- `diffusiondet/distillation.py`：严格裁剪加载、Bernoulli KL、GT 对齐/质量门控。
- `diffusiondet/detector.py`：显式训练教师路径，默认无教师路径保持原损失。
- `scripts/v3_distill.py`：配对训练、完整评测、输入/权重校验、逐步 JSONL 与最终检查点。
- `configs/lab/v3/sdd.res50.student4.yaml`：真实四级学生的基础配置；命令行固定本轮步数。
- 新增 11 项 KD 测试，与已有 15 项合计 26 passed。
- GPU 冒烟：两臂各 5 步 + 32 图评测，严格加载、输入一致、教师冻结、梯度有限均通过。
  冒烟 AP 来自类别不全的子集，不参与选型。
- 真实 SDD 权重的裁剪对拍：CPU 合成输入上，物理四级学生与六级模型的四级出口
  输出框/分数逐 bit 一致；记录在 `results/raw/v3_student_export_check.json`。
  教师/学生参数分别为 110,560,770 / 83,077,868，缩减约 24.9%。

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_distill.py \
  --gpu 1 --steps 200 --train-seeds 10 11 12 --eval-seeds 0 1 2 \
  --out-dir runs/v3/kd_pilot_new_run
```

已完成的运行目录：`runs/v3/kd_pilot_20261011/`。全部 3 对结果与判定见 RESULTS.md。
该目录 `source_snapshot/` 已保存与 manifest hash 一致的本轮源码及配置继承链。
每臂保存 `training.jsonl`、`training_summary.json`、`evaluation.json`、
`model_final.pth` 和评测原始预测。根目录保存 config、manifest、metrics.csv，
全部正常完成后写 `COMPLETE.json`。检查点仅保存学生和训练元信息，不含优化器状态；
当前脚本不支持断点恢复，重新运行须使用新的输出目录。

## 解释边界

- 无 KD 学生也可能因继续微调而提升：只有 `kd - control` 才是 KD 的附加收益。
- 同时比较 G1 原六级教师和未微调 H4，以展示最终精度/成本，但不能把该差值全部归因 KD。
- 如果 KD 不优于控制，保留负结果并检查质量门控覆盖、分类/定位损失比例及每类差异；
  不能只选择一个好 seed 宣称通过。
- 旧 G1 manifest 的源码 hash 记录当时版本；本轮增加训练路径后 hash 自然变化。
  各轮以自己的 manifest 为准，旧原始数据不重写。
