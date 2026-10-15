# R2：ROI GPU层选择与采样

协议固定于2026-10-15，数值/性能结果前记录。R1消除了batch索引构造同步，但保留每图16次
逐层nonzero同步。本轮测试GPU侧层选择：固定ROI输出容量，每个ROI只在所属FPN层执行采样，
不生成CPU可见的变长索引，不在四层重复计算所有ROI的插值。

## 实现与正确性边界

- 可选`EVAL_ROI_GPU_DISPATCH=True`；默认关闭。使用已安装的Triton，不修改Detectron2/torchvision。
  保留Torch原FPN层分配公式及R1已知数量batch索引，GPU kernel只负责按层选择和ROIAlign采样。
- 专门支持CUDA、连续FP32特征、ROIAlignV2、sampling_ratio=2、无外层autocast、eval+no_grad。
  训练、有梯度调用、CPU、tracing及不支持的pooler设置使用原ROIPooler；不实现新反向kernel。
- 保留列表、层数、batch尺寸检查；每个ROI最多在一层采样，输出仍为K×C×7×7。
  报告额外输出/临时内存、kernel启动数；不能把消除同步直接当作速度收益。
- 合成测试覆盖不等长/空batch、空层、FPN分界、越界/退化ROI及非连续特征回退。
  GPU pooled feature与native ROIAlign使用rtol=1e-5、atol=2e-5；不得失败后放宽容差。
  回退路径输出和CPU梯度精确一致；CUDA训练回退梯度rtol/atol=2e-5。
- 先执行合成数值和完整模型冒烟。若不通过，停止本候选，记录原因，不启动正式速度选型。

## 预算与比较

两臂：R1 known-count pooler控制 / GPU dispatch候选，隔离剩余nonzero路径；不叠加B1。
使用T1三个600步uniform H4末步权重，P300、NFE1、原focal、no_weights、FP32，
train seed10/11/12×eval seed0/1/2×两臂，18次完整750图SDD开发验证。
仅GPU1、4 CPU线程，TF32/cuDNN benchmark/cuDNN deterministic关闭，共享300框噪声、top-k300。
旧权重只读；主配置的原pooler默认不变。

20图预热+200图同步模型调用计时，并另做9组逐图AB/BA交错计时。
包含噪声生成、完整模型、层分配和GPU采样；不含CPU读取/resize、设seed、evaluator/数值审计。
JIT编译必须在计时前完成，单列冷启动成本与缓存位置，不把编译计入稳态速度，也不隐去它。
计时阶段不使用hooks/profiler。全部计时后做3份权重×2臂×10图profiler，验证nonzero和ROI域同步。

## 预设门槛

1. 全部数值/梯度/有限性检查通过，权重起止状态不变；控制总体/逐类AP精确复现T1。
2. 候选平均ΔAP和ΔAP75均≥−0.05，每份checkpoint平均ΔAP≥−0.1，最差类别平均ΔAP≥−0.3。
   浮点算子不同，不能预先宣称最终预测逐bit相同；完整AP与特征误差分别报告。
3. 相对R1控制，交错p50平均降低≥5%、每份checkpoint≥3%，平均p95增幅≤5%。
4. ROI域nonzero消失，ROI域同步降低；同时报告kernel数和峰值allocated显存。

先在checkpoint内平均推理seed，再跨checkpoint等权平均；速度降幅平均checkpoint比值。
若通过，也仅列为当前硬件/开发集执行候选，不自动改变默认、不宣称训练或精度增强。
失败不改门槛、不追加precision/P值搜索。所有产物保留于新的`runs/v3/`目录。
