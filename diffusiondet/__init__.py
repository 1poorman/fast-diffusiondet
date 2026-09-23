 # ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_diffusiondet_config
from .detector import DiffusionDet
# NOTE: upstream experimental `detector_dpm3.py`（Dpm3Det）已归档至 legacy/experimental/。
#       它存在必崩缺陷（缺少 multistep_predictor_update / ddim_sample，
#       self.noise_schedule 未赋值），本项目从 diffusiondet/solvers/ 重新实现，请勿回引。
from .dataset_mapper import DiffusionDetDatasetMapper
from .test_time_augmentation import DiffusionDetWithTTA
from .swintransformer import build_swintransformer_fpn_backbone
