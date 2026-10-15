# V3 工作状态

更新：2026-10-15。用户已授权继续验证，并在本轮完成后按功能分批提交。

| 项目 | 状态 |
|---|---|
| v1/v2 设计、实验和实现审计 | 完成：NFE=1 内仍有 6 级头；旧最佳 ckpt 缺失；计时/方差解释需收紧 |
| 技术检索 | 完成：D-FINE、DEIM、RT-DETRv3、DEIMv2、RF-DETR、EdgeCrafter |
| 新蓝图 | 完成，见 BLUEPRINT.md；准确区分已实现功能和后续 KD/分布回归/骨干路线 |
| 可配置推理深度与 no_weights 注意力 | 已实现，默认保持原路径；严格兼容旧权重 |
| G0 数值/梯度/跳层验证 | 4 个新测试通过；11 个旧 DDIM/NFE/求解器回归通过 |
| GPU 冒烟 | 32 图 × 2 臂完成，仅检查管线，不能作为 AP 结论 |
| G1 全验证 | 已完成：750 图 × 5 臂 × 3 推理 seed；原始数据和图已归档，Gate 未通过，见 RESULTS.md |
| G2 精度补偿训练 | 已完成 3 对 × 200 步、18 次全量评测；证据不足，不晋级默认模型，见 RESULTS.md |
| T1 训练时刻对齐 | 已完成3对×600步、18次全量评测；AP+0.465但AP75−0.160，不晋级默认；47项测试通过 |
| 冻结预测的误差诊断 | 完成：H4更多中低分背景框，固定阈值GT覆盖率未一致下降；详见RESULTS |
| P1 固定候选框预算 | 完成36次全量评测；小P均未过门槛，提速不足1%、AP下降；保持P300 |
| Q1 质量感知分类与同预算H6 | 完成6个新模型、27次全量评测；AP−0.170、AP75−0.720，保持focal；同预算H4−H6为−0.278 AP、p50低17.76% |
| L1 冻结模型耗时审计 | 完成27组计时、9组算子轨迹；5490次输出对拍通过；ROI同步不随P300→100减少，详见RESULTS |
| B1 仅骨干FP16推理 | 完成18次全验证、9组交错计时；ΔAP+0.003，p50降低11.66%未达15%门槛，保持FP32默认 |
| R1 ROI已知数量索引 | 完成18次全验证、9组交错计时及6组profiler；预测精确一致，ROI同步28→16次/图，p50降低3.26%未达5%门槛 |

## 产物

- `runs/v3/smoke_20261011/`：冒烟。
- `runs/v3/head_sweep_20261011/`：主实验，逐臂写入 CSV；包含配置、权重/代码/标注 hash、预测。
- `scripts/v3_summarize.py`：完成后归档 CSV、逐类指标、manifest，并生成 Pareto 图。
- 可追踪的小型实验结果归档到 `results/raw/v3_head_sweep*`；大预测和日志留在忽略目录 `runs/`。

本轮只使用 GPU1，没有更改/删除历史权重，没有占用 GPU0/2/3。

## G1 GPU 快照与结束状态

启动前：GPU0/1/2/3 显存分别约 3184/0/21901/21901 MiB；只绑定 GPU1。
运行中：GPU1 约 1083 MiB、54% 利用率，其他卡任务未干预。
主验证进程正常退出（exit code 0），没有留在后台的训练任务。
两份新 YAML 配置均通过解析；Python 语法、文档相对链接和 git diff 空白检查通过。

## G2 继续工作

已完成：`runs/v3/kd_pilot_20261011/`；冒烟：`runs/v3/kd_smoke_20261011/`。
重新查询 GPU1 空闲后仅绑定 GPU1。学生实际裁剪至四级，教师只用于训练。
G2 的实际协议、损失与配对校验见 [DISTILLATION.md](DISTILLATION.md)。
26 项测试通过，真实权重四级裁剪输出与原四级出口逐 bit 一致；三对输入/初始化、教师冻结检查通过。
主进程 exit code 0，`COMPLETE.json` 存在，原始 CSV/配对汇总/逐类指标/图已归档。
结束后查询 GPU1 显存为 0 MiB；没有遗留训练任务。
下一优先级调整为稳定学生微调预算与单独的训练时刻对齐实验；当前 KD 保留可选，未成为默认。

## T1 继续工作

协议已固定于 [TIME_ALIGNMENT.md](TIME_ALIGNMENT.md)：真实H4、无KD、uniform/mix_tmax，
每臂600步，训练seed10/11/12，推理seed0/1/2。完整同预算对照隔离 t 分布收益。
实现：`diffusiondet/time_sampling.py`、`scripts/v3_time_alignment.py`；默认 hook 关闭。
CPU 38 项测试通过，包括空GT、补齐、截取、默认uniform兼容、分布端点和经验频率。
GPU1启动前空闲；冒烟目录 `runs/v3/time_smoke_20261011/`。
冒烟已正常退出，2臂×3步、16图评测与全部配对审计通过。
主实验目录 `runs/v3/time_alignment_20261011/`，已正常完成，`COMPLETE.json` 存在。
另有6项结果审计测试、3项错误分类/几何/相关性测试通过，累计47项。
冻结G1预测的误差诊断已归档到 `results/raw/v3_error_audit_*`，不影响T1训练和选型门槛。
主结果已归档到 `results/raw/v3_time_alignment*` 和 `results/figures/v3_time_alignment.png`。
三对完整随机数/噪声/输入/学习率审计、6份权重与12份源码快照校验通过。
全测试集47 passed；训练进程exit code 0；结束后GPU1显存0 MiB，没有遗留训练任务。
保持uniform训练分布：T1总体AP通过但AP75未通过预设门槛。600步H4 uniform为后续控制基础，
下一步同预算H6对照与独立质量监督；不继续搜索mix概率，也未宣称分布回归已实现。

## P1 继续工作

接续2026-10-12已完成的`runs/v3/proposal_smoke_20261012/`，本轮正式目录为
`runs/v3/proposal_sweep_20261013/`。GPU1启动前0 MiB，继续只绑定GPU1。
使用T1三份600步uniform H4权重，P300/200/150/100、推理seed0/1/2、750图，
共36次评测；固定共享300框噪声池和pre-NMS top-k300，不重新训练。
新增`scripts/v3_summarize_proposals.py`：核验完整矩阵、逐级覆盖计数、
P300与T1逐项AP一致、权重/源码/标注哈希，再按预设门槛判定。
T1记录原始标注哈希，P1记录去掉background类别的评测标注哈希；
汇总分别验证两份文件，并核验过滤后内容一致，不能直接比较这两个不同文件的哈希。
24项检测头/采样器/候选预算测试、32项KD/时刻/诊断测试和17项P1汇总审计测试通过。
正式运行已正常退出，`COMPLETE.json`存在；36行评测、1296行覆盖率、九组P300全指标
复现（最大AP偏差0）、六张真实图逐bit对拍、三份权重与源码快照核验通过。
结果已归档至`results/raw/v3_proposal_sweep*`，图及结论见RESULTS。
P200/150/100分别损失0.303/0.457/0.999 AP，p50仅下降0.50%/0.71%/0.93%，
均未通过预设门槛。保持P300，不因P200接近AP底线而放宽门槛。
结束后GPU1显存0 MiB。用户已授权将当前改动按功能分批提交；大型运行产物保持忽略。
提交按依赖拆为检测头、蒸馏、时刻对齐、误差诊断、P1和总览文档六批。
另有一笔归档修复，恢复首次暂存时已被Git规范化的三份G1 CSV换行字节，不改变数值。
V3原始CSV保留原始换行字节；P1数据与实验SHA256一致，全部V3 CSV按提交内容逐字节核验。

## Q1 质量感知分类与同预算H6

用户继续其他方向后，固定协议于[QUALITY_SUPERVISION.md](QUALITY_SUPERVISION.md)。
只替换H4最后一级分类监督为带原alpha的quality focal，GT目标为detached匹配IoU，
原Dynamic-K、辅助头、L1/GIoU和全部推理结构保持不变。默认focal，旧权重严格兼容。
新训练H4 quality和H6 focal，每臂600步、seed10/11/12；复用T1三份H4 uniform控制并
重新全量评测，逐步核对新训练的输入、填充、噪声、时刻、全局RNG和LR。
冒烟`runs/v3/quality_smoke_20261013/`正常退出，2个新臂×3步+3臂×16图评测完成；
控制首批损失最大偏差约2e-7，H4初始化和H6前四级初始化、配对训练记录检查通过。
83项全量测试和18项新汇总审计测试通过，累计101项。
正式目录`runs/v3/quality_20261013/`，仅GPU1，6个新模型各600步、27次750图评测全部完成。
原始协议/源码及T1参考日志已快照；所有新产物位于`runs/v3/`，不覆盖旧权重。
主进程exit code 0，`COMPLETE.json`存在；结束后GPU1显存0 MiB。
九份模型、初始化、源码快照、标注与T1参考记录哈希核验通过；新臂逐步配对全部一致。
三次原focal首批重放通过；九组H4控制重新评测的全部AP指标与T1完全一致（最大偏差0）。
结果与描述性误差诊断已归档至`results/raw/v3_quality*`，图为`results/figures/v3_quality.png`。
Q1平均AP−0.170、AP75−0.720，仅1/3训练配对AP为正，延迟+0.037%；未通过预设门槛。
score≥0.3时背景框少41个但正确匹配也少38个，不能只报告背景抑制而忽略覆盖损失。
同600步预算，H4/H6原focal分别67.777/68.056 AP、28.087/34.152ms；
H4的p50低17.76%，AP少0.278，未达到联合目标，仍需独立数据确认。
保持uniform、P300、原focal；不按单个seed选型，不追加损失超参数搜索。
本轮提交按质量损失、配对实验与结果、总览文档三批；大型权重与预测保持忽略。

## L1 计算成本与同步诊断

Q1完成后继续验证执行方向，协议固定于[LATENCY_PROFILE.md](LATENCY_PROFILE.md)。
不训练、不计算子集AP；使用T1三个H4与Q1三个H6原focal末步权重，保持FP32、P300/P100、
共享噪声和固定top-k。实际GPU1运行目录`runs/v3/latency_20261014/`。
先完成27个单元各20图预热+200图的无插桩/嵌套event计时，再运行9组×10图profiler，
避免CUPTI影响后续基准。临时包装在退出/异常时恢复，不修改生产前向或安装依赖。
修订后的冒烟`runs/v3/latency_smoke_isolated_20261014/`正常退出；早期冒烟只检查管线，
其profiler与计时交错的数据不作最终性能依据。

正式进程与汇总均exit code 0，`COMPLETE.json`存在；GPU1已回到0 MiB。
完整矩阵、逐图区间/分位数、执行H/P、初始化来源、六份权重/源码/标注/轨迹哈希核验通过。
5400次event输出对拍与90次profiler输出对拍逐字节一致；13项新增审计测试通过，
连同已有检测头/候选预算回归共26项通过。最后新增的同步算子归属检查也通过13项审计测试。
汇总归档为`results/raw/v3_latency*`和`results/figures/v3_latency.png`；大轨迹与逐图记录保留于run目录。

无插桩H4/P300、H4/P100、H6/P300的p50均值分别28.645/27.473/33.238ms。
减框平均降幅4.10%，单轮差值波动较大；不改写P1原始计时与失败判定。
ROI每图同步28/28/42次；H4/P300的backbone/head event区间约41%/48%，attention约3.7%。
后续优先分别检验骨干精度模式和ROI动态长度同步；继续保持原focal、uniform及P300。
按工具与协议、归档结果与总览文档两批提交；没有生成新模型或覆盖历史权重。

## B1 仅骨干FP16推理

接续未完成的精度模式实现，协议固定于[BACKBONE_PRECISION.md](BACKBONE_PRECISION.md)。
默认FP32；可选模式仅对backbone/FPN启用CUDA autocast，特征转回FP32再进入head。
训练路径和checkpoint键保持兼容。使用T1三个600步uniform H4末步权重，不重新训练。
冒烟`runs/v3/backbone_precision_smoke_20261015/`已完成：seed10、推理seed0、32图，两臂dtype和冻结状态检查通过。
正式目录`runs/v3/backbone_precision_20261015/`：18次750图验证、9组逐图交错计时完成。
仅GPU1；正式进程与汇总exit code 0，`COMPLETE.json`存在，结束后显存0 MiB。

九组FP32总体/逐类AP完整复现T1，最大偏差0。13,500次验证图像调用最终预测均有限，
12条dtype审计确认骨干FP16计算、head及最终输出FP32；三份权重起止digest不变。
源checkpoint自带12个FP64扩散日程buffer，审计确认其保留原样，没有误判为权重精度变化。
17份源码快照、三份权重、标注、T1参考与81份预测/计时产物hash、逐图顺序和分位数全部通过。
原始小型数据归档至`results/raw/v3_backbone_precision*`，图为`results/figures/v3_backbone_precision.png`。

FP32/骨干FP16为67.777/67.780 AP、74.103/74.117 AP75；交错p50为27.815/24.570ms。
平均p50降低11.66%，三份checkpoint为11.45%/11.73%/11.82%，未达到平均15%门槛。
其余精度和p95条件通过；FP16保留可选，不晋级本轮部署候选，继续保持FP32默认。
下一优先级为独立ROI动态长度同步优化，不叠加QFL/减框，也不改本轮门槛。
27项新增精度边界/结果审计测试通过；全测试集141 passed，包含旧求解器和训练回归。
按推理模式、配对实验与归档、总览文档分批提交，大型运行产物保持忽略。

## R1 ROI已知数量索引

协议固定于[ROI_KNOWN_COUNTS.md](ROI_KNOWN_COUNTS.md)。本地pooler子类仅替换batch索引构造，
保留原FPN层分配、nonzero、ROIAlign及输入检查；默认仍使用原pooler，不修改安装的Detectron2。
13项新增pooler/完整head/输入验证测试、19项结果审计测试通过；CUDA输出精确一致，
冒烟feature梯度最大绝对差7.45e-9。32图冒烟`runs/v3/roi_known_counts_smoke_20261015/`已完成，
预测逐bit一致，ROI同步从每图28次降至16次，repeat_interleave调用消失。
正式目录`runs/v3/roi_known_counts_20261015/`，仅GPU1，使用T1三份600步uniform H4末步权重，
全模型FP32、P300；18次全验证、9组逐图交错计时，全部计时后再做6组profiler。
正式进程与汇总已exit code 0，`COMPLETE.json`存在；结束后GPU1显存0 MiB。
6,750对全验证图像预测精确一致，全部总体/逐类AP复现T1；3,960次交错调用和60次profiler调用对拍通过。
三份模型状态不变，21份源码、权重、标注、T1参考与105份预测/计时/轨迹产物hash核验通过。
六组正式profiler确认ROI同步28→16次/图、repeat_interleave消失；安装的Detectron2源码未变。
完整测试集173 passed。原始结果归档至`results/raw/v3_roi_known_counts*`，图为`results/figures/v3_roi_known_counts.png`。
两臂同为67.777 AP、74.103 AP75；交错p50为27.494/26.598ms，平均降低3.26%，未达预设5%门槛。
每checkpoint降幅3.23%/3.31%/3.23%，p95平均降低3.35%；其他条件通过。
保持原pooler默认，known-count模式保留可选，不与B1叠加宣称通过。
下一步可独立验证剩余逐层nonzero路径，必须先核验数值/梯度与额外计算开销。
按pooler实现、实验与归档、总览文档分批提交，大型运行产物保持忽略。

## R2 GPU侧ROI层选择与采样

协议固定于[ROI_GPU_DISPATCH.md](ROI_GPU_DISPATCH.md)。Triton kernel按原FPN层分配只采样所属
层，沿用R1已知框数量索引，不创建逐层CPU可见变长索引。默认关闭；限定CUDA连续FP32、
四层ROIAlignV2、sampling_ratio=2、eval+no_grad；训练/梯度、CPU、tracing、autocast及不支持设置
回退原ROIPooler，不修改安装的Detectron2/torchvision，也不实现新反向算子。

早期Triton 2.1启动/除法API和数据路径错误均保留于独立run目录；修复后未调整数值容差。
最终前置目录`runs/v3/roi_dispatch_prerequisite_20261015/`：合成最大误差1.19e-7，
32图/128次真实ROI调用最大误差3.81e-6，固定rtol=1e-5、atol=2e-5全部通过；
CUDA原生梯度回退误差0，非连续特征/autocast输出精确一致。
管线冒烟`runs/v3/roi_gpu_dispatch_smoke_20261015/`完整退出，计时期间没有新编译。

正式目录`runs/v3/roi_gpu_dispatch_20261015/`，仅GPU1、FP32/P300，T1三份600步H4权重只读。
18次750图完整评测、9组逐图交错计时、全部计时后6组profiler完成；进程和汇总均exit code 0，
`COMPLETE.json`存在，结束后GPU1显存0 MiB。输入形状预编译准备完成，25份缓存文件在每份
checkpoint计时期间不变，结束后hash验证通过；256通道/300框空缓存首次调用约1.498秒。

控制九组总体/逐类AP精确复现T1，13,500次验证图像调用预测有限；两臂AP均约67.777，
平均ΔAP+0.00000232、ΔAP75=0，最差类别平均ΔAP=0。最终预测digest均不逐bit相同，
不把AP接近描述成预测精确一致。3,960次交错调用和60次profiler调用精确复现各自臂输出。
三份模型起止state不变，24份源码、三份权重、标注/T1参考和109份产物hash审计通过。

交错p50为27.238→20.194ms，平均降低25.87%；三份checkpoint降幅26.35%/26.13%/25.13%，
p95平均降低26.18%，通过全部预设准确率/速度门槛。ROI nonzero和stream同步16→0次/图，
ROI kernel启动188→64次/图；模型峰值allocated显存均0.576337GiB，未观察到增加。
结果归档至`results/raw/v3_roi_gpu_dispatch*`和`results/figures/v3_roi_gpu_dispatch.png`。
8项新增pooler/完整head回归、29项结果审计测试通过，完整测试集210 passed。

R2列为当前RTX 3090 / SDD开发集可选执行候选，默认仍为原pooler、FP32、uniform、P300和原focal。
下一步优先独立数据与目标硬件确认；若与B1组合，需另立协议，不能相加不同运行轮次的速度收益。
按推理实现、实验与归档、总览文档分批提交；大型预测、轨迹和编译缓存保持忽略。
