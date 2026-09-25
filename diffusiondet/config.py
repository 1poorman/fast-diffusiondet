# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os

from detectron2.config import CfgNode as CN


def add_diffusiondet_config(cfg):
    """
    Add config for DiffusionDet
    """
    cfg.MODEL.DiffusionDet = CN()
    cfg.MODEL.DiffusionDet.NUM_CLASSES = 80
    cfg.MODEL.DiffusionDet.NUM_PROPOSALS = 300

    # RCNN Head.
    cfg.MODEL.DiffusionDet.NHEADS = 8
    cfg.MODEL.DiffusionDet.DROPOUT = 0.0
    cfg.MODEL.DiffusionDet.DIM_FEEDFORWARD = 2048
    cfg.MODEL.DiffusionDet.ACTIVATION = 'relu'
    cfg.MODEL.DiffusionDet.HIDDEN_DIM = 256
    cfg.MODEL.DiffusionDet.NUM_CLS = 1
    cfg.MODEL.DiffusionDet.NUM_REG = 3
    cfg.MODEL.DiffusionDet.NUM_HEADS = 6

    # Dynamic Conv.
    cfg.MODEL.DiffusionDet.NUM_DYNAMIC = 2
    cfg.MODEL.DiffusionDet.DIM_DYNAMIC = 64

    # Loss.
    cfg.MODEL.DiffusionDet.CLASS_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.GIOU_WEIGHT = 2.0
    cfg.MODEL.DiffusionDet.L1_WEIGHT = 5.0
    cfg.MODEL.DiffusionDet.DEEP_SUPERVISION = True
    cfg.MODEL.DiffusionDet.NO_OBJECT_WEIGHT = 0.1

    # Focal Loss.
    cfg.MODEL.DiffusionDet.USE_FOCAL = True
    cfg.MODEL.DiffusionDet.USE_FED_LOSS = False
    cfg.MODEL.DiffusionDet.ALPHA = 0.25
    cfg.MODEL.DiffusionDet.GAMMA = 2.0
    cfg.MODEL.DiffusionDet.PRIOR_PROB = 0.01

    # Dynamic K
    cfg.MODEL.DiffusionDet.OTA_K = 5

    # Diffusion
    cfg.MODEL.DiffusionDet.SNR_SCALE = 2.0
    cfg.MODEL.DiffusionDet.SAMPLE_STEP = 1

    # Inference
    cfg.MODEL.DiffusionDet.USE_NMS = True

    # ---------------- M4: EDM training formulation (F1) ----------------
    # 训练范式：vp（原 VP-DDPM）| edm（Karras 预条件 + LogNormal σ + λ(σ) reg 加权）
    cfg.MODEL.DiffusionDet.FORMULATION = "vp"
    # EDM σ 采样：sigma ~ exp(N(P_MEAN, P_STD))，截断 [SIGMA_MIN, SIGMA_MAX]
    cfg.MODEL.DiffusionDet.P_MEAN = -1.2
    cfg.MODEL.DiffusionDet.P_STD = 1.2
    cfg.MODEL.DiffusionDet.SIGMA_MIN = 0.01
    cfg.MODEL.DiffusionDet.SIGMA_MAX = 4.0
    # sigma_data：留空则读 diffusiondet/sigma_data.json 的 gt_only_global
    cfg.MODEL.DiffusionDet.SIGMA_DATA = 0.0
    # λ(σ) 加权范围：reg（A4 默认，L1+GIoU）| all（cls 也乘，不推荐）
    cfg.MODEL.DiffusionDet.EDM_LOSS_SCOPE = "reg"
    # λ(σ) 封顶：σ→0 时 λ~1/σ² 发散，对 L1+GIoU 梯度爆炸，必须封顶
    cfg.MODEL.DiffusionDet.EDM_LAMBDA_MAX = 50.0
    # M4 诊断：c_noise=lnσ/4∈[-1.15,0.35]，与迁移权重的 t∈[0,999] 嵌入差 3 个量级。
    # 缩放后 σ=exp(4·t/scale)；1.0=原 EDM（H4 否定配置），250≈对齐 VP 域（诊断）
    cfg.MODEL.DiffusionDet.EDM_CNOISE_SCALE = 1.0
    # 时间嵌入输入：c_noise = ln(σ)/4（A6 默认 Sinusoidal + 该缩放）
    # Karras ρ（E1-HEUN 采样网格）
    cfg.MODEL.DiffusionDet.KARRAS_RHO = 7.0

    # ---------------- M1: sampler abstraction (diffusiondet/solvers/) ----------------
    # 推理采样器：ddim | euler | heun | dpm_v3
    cfg.MODEL.DiffusionDet.SOLVER = "ddim"
    # DPM-Solver-v3 阶数（1~3）；SAMPLE_STEP 兼作全部 solver 的 NFE 预算
    cfg.MODEL.DiffusionDet.ORDER = 3
    # DPM-Solver-v3 时间步策略：logSNR | time_uniform | time_quadratic | edm
    cfg.MODEL.DiffusionDet.SKIP_TYPE = "logSNR"
    # Heun 二阶校正的步长守卫：|Δw| = |σ/α| 差超过该值的区间退回一阶
    # （cosine 调度末端 w~2e4，不设守卫二阶校正会灾难性发散）
    cfg.MODEL.DiffusionDet.HEUN_MAX_DW = 1.0
    # True: l=1,s=0,b=0（≈DPM-Solver++），不需要 EMS 统计量
    cfg.MODEL.DiffusionDet.DEGENERATED = True
    # EMS 统计量目录（M3 产出 l.npz/sb.npz；留空且 DEGENERATED=False 会报错）
    cfg.MODEL.DiffusionDet.STATS_DIR = ""
    # Box renewal（仅 DDIM 支持；FEP 主矩阵要求 False）
    cfg.MODEL.DiffusionDet.BOX_RENEWAL = True
    # 多步 ensemble + NMS（FEP 主矩阵要求 False）
    cfg.MODEL.DiffusionDet.USE_ENSEMBLE = True

    # ---------------- fast-diffusiondet extensions (M0) ----------------
    # 迁移初始化（决策 D2）：留空则由 diffusiondet/weights.py 自动搜索
    cfg.MODEL.DiffusionDet.PRETRAIN_PATH = os.environ.get("FASTDD_PRETRAIN", "")
    # 未被迁移覆盖的层（主要是分类头）的初始化确定性
    cfg.MODEL.DiffusionDet.CLS_REINIT_SEED = 0
    # A11：冻结 stem+res2 以省显存，默认关闭
    cfg.MODEL.DiffusionDet.FREEZE_BACKBONE_STAGE1 = False

    # 数据集根目录（PLS/SDD）；留空则 diffusiondet/data_register.py 自动搜索
    cfg.DATASET_ROOT = os.environ.get("FASTDD_DATASET", "")

    # Swin Backbones
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.SIZE = 'B'  # 'T', 'S', 'B'
    cfg.MODEL.SWIN.USE_CHECKPOINT = False
    cfg.MODEL.SWIN.OUT_FEATURES = (0, 1, 2, 3)  # modify

    # Optimizer.
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0

    # TTA.
    cfg.TEST.AUG.MIN_SIZES = (400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800, 800)
    cfg.TEST.AUG.CVPODS_TTA = True
    cfg.TEST.AUG.SCALE_FILTER = True
    cfg.TEST.AUG.SCALE_RANGES = ([96, 10000], [96, 10000], 
                                 [64, 10000], [64, 10000],
                                 [64, 10000], [0, 10000],
                                 [0, 10000], [0, 256],
                                 [0, 256], [0, 192],
                                 [0, 192], [0, 96],
                                 [0, 10000])
