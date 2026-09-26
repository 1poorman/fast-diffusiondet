results/figures 图表说明
========================
生成日期：2026-09-26
生成脚本：scripts/make_figures.py（fig1~fig3）、scripts/visualize_compare.py（vis/）
数据来源：results/raw/*.csv（评测矩阵）、output/lab/*/metrics.json（训练曲线）
模型：全部基于 E0-CTRL（SDD D4 bs8 重训，1750 iter，AP 67.16@ddim NFE=1），
     除 m5_diag 系列使用 EDM 诊断模型（仅用于验证 c_noise 假设，非正式结果）。

------------------------------------------------------------------

fig1_ap_nfe_curves.png
----------------------
左图：AP vs NFE 曲线（4 求解器 × NFE 1~10，误差棒 = 3 seed 标准差）。
右图：AP vs latency 权衡图（点标注对应 NFE，latency 为逐图 forward 中位数，
     eval bs=32 口径）。
读图要点：
  * DDIM（灰）斜率最陡：多步重注入噪声使 AP 单调下降（66.95@1 → 63.39@10）。
  * DPM-Solver++（红）在 NFE=2 有唯一"甜点"（67.11，全局最高 AP），
    随后与 DDIM 同步下滑。
  * Heun/Euler 退化最平缓（NFE=10 仍 65+），适合需要多步精修的场景。
  * 右图中 DPP@2 位于帕累托前沿"高 AP 端"，DDIM@1 位于"低延迟端"，
    其余全部配置被这两点的连线支配。

fig2_training_loss.png
----------------------
三个数据集（SDD/PLS/WHEAT）D4 训练的 total loss 曲线（浅色=原始 20 点日志，
深色=窗口 20 平滑），红色虚线竖线 = EVAL 点，顶部红字 = 该次 eval 的 COCO AP。
读图要点：
  * 三者形态一致：快速下降 → LR 衰减台阶 → 平台收敛，均无过拟合
    （eval AP 与 train loss 同步改善）。
  * PLS 曲线在 ~2400 iter 有轻微上翘后回落，属 STEPS 衰减后正常波动。
  * 注意 y 轴量级不同（数据集难度/类别数不同），不可横向比较绝对值；
    图内相对趋势才是信息。

fig3_d4_strategy.png
--------------------
D4 训练策略收益对照（SDD）：v1.0 配置（bs2/LR2.5e-5/3500it，灰）vs
D4 配置（bs8/LR1e-4/1750it，红）。x 轴统一为"看过图像数"（iter × batch size），
可直接比较数据效率。
读图要点：
  * 红曲线（D4）用一半的图像量降到更低 loss：~1600 图时已达灰曲线
    7000 图的水平——大 batch + 大 LR 的优化效率优势。
  * 终点 AP：D4 67.16 vs v1.0 61.83（+5.2，全项目最大单项收益）。
  * 灰曲线在 ~5600 图处的台阶是 LR 衰减（2800 iter）。

vis/grid_det.jpg
----------------
检测效果图对比（3 张 val 图 × 4 配置）：
  行（配置）：ddim steps=1 / ddim steps=2 / dpm_v3 steps=2 / heun steps=3
  列（图）：3 张 SDD val 图（同 seed=42，同图横向可比）
画框规则：score > 0.5 的预测框（detectron2 Visualizer）。
读图要点：
  * 高置信度大目标四配置基本一致（模型收敛良好）。
  * 差异在低置信度/小目标：ddim steps=2 开始丢小目标（与 AP -0.75 一致），
    dpm_v3 steps=2 保留更完整。
  * 注意 top-left 图目标极少属正常（该图 GT 本就只有 1 个框）。

vis/grid_cam.jpg
----------------
Grad-CAM++ 热力图对比（同 3 图 × 4 配置，与 grid_det 一一对应）。
实现：对该配置最终去噪状态 x_start 跑 head（带梯度），以最后一层类别
logits 之和为目标反传到 backbone P5（stride 32），Grad-CAM++ 加权。
读图要点：
  * ddim@1 与 dpm_v3@2 的热力图均聚焦病斑/虫害区域（与 GT 框吻合），
    说明两种配置的 backbone 注意力都正确。
  * ddim@2 行出现块状/条带伪影——重注入噪声扰动最终提议分布，
    与该配置 AP -2.8 定量一致。
  * heun@3 行最平滑（二阶 ODE 轨迹更稳定）。
  * 技术注：FPN 的 p5 是动态输出，模块级 backward hook 对 dict 输出
    不触发，故在 forward hook 里对 p5 张量直接 register_hook（仅带梯度
    前向注册）；热力图上采样到原图用双线性，stride 32 的边界模糊属方法固有。

------------------------------------------------------------------

配套数据（results/raw/）：
  m4_e0ctrl_full.csv     fig1 的数据（75 evals）
  m3_k256_scalar/percoord.csv  K=256 EMS 真统计量结果（RESULTS §5.4）
  m6_pls_*.csv           PLS 确认（天花板 98.3，solver 无差异）
  m7_wheat_all.csv       WHEAT 三配置（DPP@2 24.08 vs ddim@1 24.16，同模式）
  m5_diag_heun_fixed2.csv  c_noise 缩放诊断 + edm 修复验证

口径提醒：
  * 所有 AP 为 COCO bbox AP（IoU 0.5:0.95），3 seed 均值±std。
  * latency 为 eval bs=32 下逐图 forward 中位数；bs 不同则绝对值不可比，
    但 Δlat 百分比与 NFE 的线性关系可跨口径复用。
  * WHEAT/PLS 的训练-评测均基于各自 D4 配置（见 configs/lab/*bs8*.yaml）。
