# legacy/experimental —— 上游遗留实验代码（只读存档）

本目录的文件**全部来自 upstream fork 点**，**不可运行**，已从 `diffusiondet/` 主包中解除引用
（见 `diffusiondet/__init__.py` 的注释）。**保留仅为追溯与移植参考，请勿 import。**

| 文件 | 状态 | 已知缺陷 |
|---|---|---|
| `detector_dpm3.py` | ❌ 必崩 | ① L624/641 调用 `self.multistep_{predictor,corrector}_update`，但 `Dpm3Det` 未持有 `DPM_Solver_v3` 实例<br>② L738 `forward()` 调 `self.ddim_sample(...)`，该类无此方法<br>③ L426-467 `self.noise_schedule` 从未赋值，访问 `.total_N` 必崩 |
| `detector_noise.py` | ⚠ 冗余 | `detector.py` 的算法等价副本（多了韩文注释），未被 `__init__.py` 导入 |
| `dmp3.py` | ❌ 残缺 stub | 类体到第 30 行止，无 `forward`；`cfg.MODEL.DIFFUSIONDET` 大小写错误（实际是 `MODEL.DiffusionDet`） |
| `train.py` | ⚠ 硬编码 | 内含写死的 `E:\Files\DL code\datasets\wgisd` 路径 + WGISD 8 类数据集注册；官方入口请用根目录 `train_net.py` |
| `samplers/` | ⚠ 参考 | DPM-Solver-v3 官方实现副本，缺 `__init__.py`；**内含 CIFAR-10 统计量** |
| `samplers/l.npz`、`sb.npz` | ❌ 不适用 | 形状 `(121, 3, 32, 32)`，是 CIFAR-10 图像的 EMS 统计量，与 box 数据 `(B,P,4)` 不兼容 |
| `Base-DPM3Det.yaml` | ⚠ 孤立 | `META_ARCHITECTURE: "Dpm3Det"`，指向已移除的 meta arch，无任何配置继承它 |
| `run.txt` | — | 配合上面的 `train.py` 使用的命令记录 |

## 替代方案

本项目从 **`diffusiondet/solvers/`** 重新实现采样器：

- `solvers/ddim.py` —— 与原 `detector.py:186-274` 行为逐 bit 等价
- `solvers/heun.py` —— EDM 风格 Heun 二阶 + Karras ρ=7 时间步
- `solvers/dpm_solver_v3.py` —— DPM-Solver-v3（含 **自算** 的 box 版 EMS 统计量）

移植时必须对抗的 API 陷阱清单见 `docs/BLUEPRINT.md` §3.4。
