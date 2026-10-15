# W1: WHEAT rebuild and GPU ROI dispatch confirmation

Protocol fixed before W1 training or validation. The historical WHEAT weights are
absent. Rebuild from `pretained/diffdet_coco_res50.pth`; historical AP24.16 is
background, not an exact reproduction target. Preserve all historical artifacts.

## Training

- Three independent H6/P300 final models, seeds 10/11/12, exactly 1,400 attempted
  updates each. First run a separate seed10 ten-attempt pipeline smoke.
- Follow `configs/lab/wheat.res50.bs8.yaml`: batch8, AdamW LR1e-4 and decay1e-4,
  linear warmup100 from factor0.01, LR times0.1 at attempt1120, global gradient
  norm clipping1.0, original multiscale resize and horizontal flip, FP16 AMP.
- Uniform training times, VP, original focal plus all auxiliary heads, no KD or
  EMA. Disable unused attention weights; native pooler in training. Four CPU
  threads, loader worker0. Seed Python/NumPy/Torch before each batch using
  `image_seed(train_seed, attempt)`; sampler has its own training seed.
- No validation during training and no best-checkpoint selection. Do not extend
  the budget to compensate AMP skips or based on validation performance. Advance
  the learning-rate schedule by attempted steps. Stop on nonfinite loss/state.
  Report all skipped steps; after the first100 attempts, skip fraction must be
  strictly below1% for training health to pass. Smoke cannot establish this gate.
- Verify source annotations and every image against D1 hashes. Record config,
  source/pretrain hashes, transfer keys, initial/final state digests, input image
  IDs/digests, step seeds, losses, LR, gradient norms, scaler values, attempts and
  actual updates. Checkpoints are new files under ignored `runs/v3/`.

## Frozen evaluation

- Use all three final checkpoints, inference seeds0/1/2, and arms `known_counts`
  (R1) / `gpu_dispatch` (R2): 18 full evaluations of all123 validation images.
- H6/P300, FP32, NFE1 DDIM, shared300-box noise, top-k300, no attention weights,
  no renewal/ensemble. TF32 and cuDNN benchmark/deterministic disabled, GPU1 only.
- Before timing each checkpoint, compare pooled features on the first32 real
  images /192 ROI calls, fixed rtol1e-5, atol2e-5. Failure prevents selection.
  Also run R2 synthetic and native gradient-fallback checks unchanged.
- Prepare every actual input shape before timing, record cold preparation and
  cache hashes, reject new compilation during timing. Record final output
  digests, finite predictions, immutable state and checkpoint hashes.
- Nine per-image AB/BA timing pairs: 20 warmup images +103 unique measured images
  per arm. No repeated images to inflate the sample count. Timing includes noise,
  H2D, preprocessing, backbone/head, NMS and postprocessing with CUDA synchronization;
  excludes loader/CPU resize, seeding, digests and evaluation. No timing hooks.
- Profile only after all timings, ten images per arm/checkpoint, 60 ROI calls
  per profile. Attribute nonzero, synchronization and kernels within ROI scopes.

## Gates and interpretation

Aggregate inference seeds within each checkpoint, then weight checkpoints equally.
Require mean delta AP and AP75 >=-0.05, every checkpoint delta AP >=-0.1,
worst category mean delta AP >=-0.3; mean p50 reduction >=5%, each checkpoint >=3%,
mean p95 increase <=5%; ROI nonzero disappears and synchronization decreases.
Training health, numerical, artifact, execution and immutable-state checks must pass.
Keep the native pooler default regardless of this experiment's outcome.

D1 found695 training /123 validation images, 12 classes covered in training, and
no exact cross-split duplicates. WHEAT validation was used historically: this is
cross-dataset execution confirmation on the same RTX3090 and H6, not an unused
independent test set, new hardware validation, or evidence about H4 accuracy.
