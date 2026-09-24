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

    # ---------------- Sampler (M1) ----------------
    # 采样器选择: 'ddim' | 'heun' | 'dpm_solver_v3'
    cfg.MODEL.DiffusionDet.SOLVER = "ddim"
    # DPM-Solver-v3
    cfg.MODEL.DiffusionDet.DPM_ORDER = 3            # 求解器阶数
    cfg.MODEL.DiffusionDet.DPM_SKIP_TYPE = "logSNR"  # logSNR | time_uniform
    cfg.MODEL.DiffusionDet.DPM_DEGENERATED = True    # True -> 退化为 DPM-Solver++（不需真 EMS，M1 用）
    cfg.MODEL.DiffusionDet.DPM_STATS_DIR = "statistics/degenerated"
    # Heun / EDM 风格
    cfg.MODEL.DiffusionDet.HEUN_SOLVER = "heun"      # euler | heun
    cfg.MODEL.DiffusionDet.HEUN_RHO = 7.0            # Karras 幂律指数

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
