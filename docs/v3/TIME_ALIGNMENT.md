# T1：单步推理的训练时刻对齐

预先固定协议：2026-10-11，主实验结果产生前记录。G2 联合蒸馏仅 +0.117 AP，
未晋级；本轮关闭 KD，只改变训练 t 分布。

## 假设与实验边界

NFE=1 推理只访问 t=999；均匀训练每张图仅 0.1% 概率访问该时刻。
将 50% 预算直接分配到 t=999，另 50% 保持 uniform，可能帮助真实四级学生适应
单步输入。它不改变部署计算图，目标是改善精度，**本身不提供额外推理加速**。
高噪声分配同时减少低噪声监督，效果可能为负，必须配对验证。

- 初始化：同一现存六级短训权重，严格裁剪后两级，物理 H4，83,077,868 参数。
- 两臂：`uniform` / `mix_tmax(p=0.5)`；KD、EMA、AMP 均关闭。
- 每臂每 seed 固定 **600 步 × batch8 = 4800 图像遍历**，约 2.74 个 SDD epoch。
- train seeds：10/11/12。偶数 seed 先 uniform，奇数先 mix_tmax，顺序预先固定。
- fresh AdamW，LR=2.5e-5，weight decay=1e-4，clip norm=1；20 步 warmup，480 步 LR×0.1。
- 两臂相同预算，不用 600 步 mix 与 200 步旧 control 估算因果收益。
- 只评测最后 checkpoint；不按验证集挑选最佳 step/seed，也不根据中途 AP 提前停止。
- 750 张完整开发验证图 × inference seeds 0/1/2，先在模型内平均，再计算 3 对训练差值。
- FP32、TF32 off、P300、NFE1、800/1333、batch1 计时；沿用 G1/G2 的20图预热和200图同步计时。

## 随机数配对

旧 `mix_tmax` 在命中分支后省略 `torch.randint`，还消耗全局 CPU 随机数。
只复用 seed 不能保证 eps/填充框/增强一致。本实验通过可选 `PairedTimeSampler`：

1. 两臂每图都执行原来的 device `randint`，生成共享 base t；uniform 的全局 RNG 路径与旧版一致。
2. mixture gate 使用私有 CPU generator，每批独立种子 `image_seed(train_seed, 'tgate:{step}')`。
3. 两臂共享 gate；mix 命中时覆盖 t=999；uniform 忽略 gate。
4. 逐批检查初始权重、增强图像/GT、包含随机填充的 x_start、eps、base t、gate、forward 后全局 RNG。
   **x_t 不要求相同**，因为它正是 t 变化的预期结果。
5. 日志保留每图实际 t、base t、gate 和 source digest；记录实际 tmax 比例，理论约50.05%/0.1%。

这些 hook 仅用于实验，默认 None，原始训练/推理行为和 state_dict 键保持不变。
未开启配对 hook 的历史 `mix_tmax` 保持旧 RNG 行为，因此复现实验必须使用新 runner。
CUDA ROIAlign backward 等操作仍不保证逐 bit 可复现；相同 RNG 状态不等于所有数值操作确定。
审计会引入少量训练同步成本，两臂均启用，训练墙钟只用于本协议内比较。

## 预先固定的筛选门槛

- 平均配对 ΔAP ≥ +0.3，至少2/3训练种子为正；配对均值 ΔAP75 ≥ −0.1。
- 同图延迟 p50 增幅 ≤3%；若超过，先按原协议复测计时，不能声称训练 t 修改增加架构成本。
- 达标只代表 SDD 开发集候选，不构成最终测试集确认；不达标保持 uniform。
- 报告每对 ΔAP、训练种子样本 SD、AP75/逐类 AP、时间分布、配对审计和训练成本。
  不把3个推理种子当成3次训练，也不以少量 small GT 的 APs 做主要决策。
- 若通过，再加入同预算 H6 对照与独立数据确认架构收益；若失败，优先转向定位质量/候选预算，
  不继续在开发验证集反复搜索混合概率。

## 复现

```bash
/home/huachenghao/.conda/envs/fastdiff/bin/python scripts/v3_time_alignment.py \
  --gpu 1 --steps 600 --train-seeds 10 11 12 --eval-seeds 0 1 2 \
  --out-dir runs/v3/time_alignment_20261011
```

小规模冒烟使用 `--steps 3 --train-seeds 10 --eval-seeds 0 --eval-limit 16`，仅验证管线。
目录必须不存在；不覆盖旧结果。每臂保存最终学生权重和完整审计，不支持中断训练续接。

## 完成状态（2026-10-12汇总）

`runs/v3/time_alignment_20261011/` 已完成6次600步训练、18次全量评测。
平均AP 67.777→68.243（+0.465），AP75 74.103→73.943（−0.160），p50约27.6ms。
AP75未满足预设−0.1底线，T1不晋级默认；保持uniform。逐对结果与证据边界见
[RESULTS.md](RESULTS.md)。复现本轮以运行目录中的预先保存协议/源码快照为准。
归档运行的runner固定使用0.1→1 warmup；原导出配置中的继承字段
`WARMUP_FACTOR=0.01` 未参与计算，实际逐步LR已完整记录并核验两臂一致。
后续runner已将导出字段同步为0.1，只纠正元信息，不改变训练算法；旧归档不重写。
