"""P1: frozen H4 checkpoints, shared 300-query noise pool, fixed pre-NMS top-k.

Evaluates all training-seed checkpoints, not a best validation seed. Constant P
is a screen before any progressive pruning or retraining, neither is claimed.
"""
import argparse
from collections import defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.v3_head_sweep import image_seed, sha256
from scripts.v3_error_audit import box_iou
from scripts.v3_summarize_distill import write_rows


def accumulate_coverage(accumulator, stage_boxes, record, input_hw):
    """Class-agnostic pre-NMS GT coverage; explicitly not COCO matched AR."""
    import numpy as np
    from detectron2.structures import BoxMode
    anns = [a for a in record["annotations"] if not a.get("iscrowd",0)]
    if not anns:
        return
    gt = np.array([BoxMode.convert(a["bbox"],a["bbox_mode"],BoxMode.XYWH_ABS) for a in anns])
    labels = np.array([a["category_id"] for a in anns])
    height, width = input_hw
    boxes = stage_boxes.copy() * np.array([record["width"]/width,record["height"]/height]*2)
    boxes[...,0::2] = boxes[...,0::2].clip(0,record["width"])
    boxes[...,1::2] = boxes[...,1::2].clip(0,record["height"])
    boxes[...,2:] -= boxes[...,:2]
    for stage, xywh in enumerate(boxes,1):
        best = box_iou(xywh,gt).max(axis=0)
        groups = {"all":np.ones(len(gt),dtype=bool)}
        if len(gt) >= 5:
            groups["crowded_ge5"] = groups["all"]
        groups.update({f"category_{c}":labels==c for c in np.unique(labels)})
        for group, mask in groups.items():
            counts = accumulator[(stage,group)]
            counts[0] += int(mask.sum())
            counts[1] += int((best[mask]>=0.5).sum())
            counts[2] += int((best[mask]>=0.75).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-file",default="configs/lab/v3/sdd.res50.student4.yaml")
    p.add_argument("--checkpoint-root",type=Path,default=Path("runs/v3/time_alignment_20261011"))
    p.add_argument("--dataset-root",default="datasets")
    p.add_argument("--gpu",default="1")
    p.add_argument("--train-seeds",type=int,nargs="+",default=[10,11,12])
    p.add_argument("--eval-seeds",type=int,nargs="+",default=[0,1,2])
    p.add_argument("--proposals",type=int,nargs="+",default=[300,200,150,100])
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--out-dir",type=Path,required=True)
    args=p.parse_args()
    if args.limit < 0 or 0 < args.limit <= 20:
        p.error("limit must be 0 (full) or greater than 20 (warmup)")
    if 300 not in args.proposals or any(not 1 <= n <= 300 for n in args.proposals):
        p.error("proposals must include baseline 300 and lie in [1,300]")
    for values in (args.train_seeds,args.eval_seeds,args.proposals):
        if len(values)!=len(set(values)):
            p.error("Seed and proposal values must be unique")
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    import numpy as np
    import torch
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog, build_detection_test_loader
    from detectron2.modeling import build_model
    from detectron2.evaluation import COCOEvaluator
    from detectron2.utils.logger import setup_logger
    from diffusiondet import add_diffusiondet_config
    from diffusiondet.data_register import register_all
    from diffusiondet.proposal_budget import proposal_noise
    from diffusiondet.util.model_ema import add_model_ema_configs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.out_dir.mkdir(parents=True,exist_ok=False)
    setup_logger(output=str(args.out_dir))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cfg=get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.DATASET_ROOT=str(Path(args.dataset_root).resolve())
    cfg.MODEL.DEVICE="cuda"
    cfg.MODEL.DiffusionDet.INFERENCE_TOPK=300
    cfg.OUTPUT_DIR=str(args.out_dir)
    cfg.freeze()
    dd=cfg.MODEL.DiffusionDet
    if (dd.NUM_HEADS!=4 or dd.EVAL_NUM_HEADS!=0 or dd.SAMPLE_STEP!=1 or dd.FORMULATION!="vp"
            or dd.SOLVER!="ddim" or dd.BOX_RENEWAL or dd.USE_ENSEMBLE or dd.ATTENTION_NEED_WEIGHTS):
        raise ValueError("P1 requires H4, NFE1 VP DDIM, no renewal/ensemble/attention weights")
    register_all(cfg=cfg)
    full_records=sorted(DatasetCatalog.get("sdd_val"),key=lambda r:r["image_id"])
    records=full_records[:args.limit] if args.limit else full_records
    loader=build_detection_test_loader(dataset=records,mapper=DatasetMapper(cfg,is_train=False),num_workers=0,batch_size=1)
    metadata=MetadataCatalog.get("sdd_val")
    (args.out_dir/"config.yaml").write_text(cfg.dump())
    sources=[Path(__file__).resolve().relative_to(Path.cwd()),Path(args.config_file),Path("docs/v3/PROPOSAL_BUDGET.md"),
             *[Path("diffusiondet")/f for f in ("proposal_budget.py","config.py","detector.py","head.py","solvers/__init__.py","solvers/ddim.py")],
             Path("scripts/v3_head_sweep.py"),Path("scripts/v3_error_audit.py"),Path("scripts/v3_summarize_distill.py")]
    for source in sources:
        dst=args.out_dir/"source_snapshot"/source
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,dst)
    checkpoints={str(s):args.checkpoint_root/f"seed{s}_uniform/model_final.pth" for s in args.train_seeds}
    manifest=dict(args={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  checkpoint_sha256={s:sha256(path) for s,path in checkpoints.items()},
                  source_sha256={str(path):sha256(path) for path in sources},
                  annotation_sha256=sha256(metadata.json_file),image_ids=[r["image_id"] for r in records],
                  full_images=len(full_records),category_names=metadata.thing_classes,
                  git_revision=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                  torch=torch.__version__,detectron2=detectron2.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
                  protocol="Frozen physical H4 T1 uniform checkpoints; FP32 TF32 off; batch1; shared 300-noise pool prefix; pre-NMS topk300; NMS unchanged; 20 warmup+200 timed images",
                  timing="Full-pool noise generation+prefix+complete model call; CUDA synchronized; excludes loader/CPU resize/evaluator/coverage computation",
                  coverage="Per-stage class-agnostic best GT IoU before NMS, boxes clipped to image; not COCO AR; crowded means >=5 GT")
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows,coverage_rows,parity_rows=[],[],[]
    for train_seed in args.train_seeds:
        model=build_model(cfg).eval()
        state=torch.load(checkpoints[str(train_seed)],map_location="cpu")
        if state.get("arm")!="uniform" or state.get("train_seed")!=train_seed:
            raise ValueError("Expected matching T1 uniform checkpoint")
        model.load_state_dict(state["model"],strict=True)
        del state
        # Same full-pool noise must reproduce normal inference bit for bit.
        with torch.no_grad():
            for i,batch in enumerate(loader):
                if i>=2:
                    break
                torch.manual_seed(image_seed(0,batch[0]["image_id"]))
                model.inference_topk=0
                reference=model(batch)[0]["instances"]
                torch.manual_seed(image_seed(0,batch[0]["image_id"]))
                model.inference_topk=300
                noise=proposal_noise(1,300,300,model.device)
                actual=model(batch,inference_noise=noise)[0]["instances"]
                for name in ("scores","pred_classes"):
                    if not torch.equal(getattr(reference,name),getattr(actual,name)):
                        raise RuntimeError(f"P300 noise injection parity failed: {name}")
                if not torch.equal(reference.pred_boxes.tensor,actual.pred_boxes.tensor):
                    raise RuntimeError("P300 noise injection box parity failed")
                parity_rows.append(dict(train_seed=train_seed,image_id=batch[0]["image_id"],bit_exact=True))
        (args.out_dir/"parity.json").write_text(json.dumps(parity_rows,indent=2)+"\n")
        holder={}
        hook=model.head.register_forward_hook(lambda module,inputs,output:holder.update(coords=output[1].detach()))
        for seed in args.eval_seeds:
            order=args.proposals if (seed+train_seed)%2==0 else list(reversed(args.proposals))
            for count in order:
                model.num_proposals=count
                arm_dir=args.out_dir/f"train{train_seed}_eval{seed}_p{count}"
                arm_dir.mkdir()
                evaluator=COCOEvaluator("sdd_val",tasks=("bbox",),distributed=False,output_dir=str(arm_dir))
                evaluator.reset()
                latencies=[]
                coverage=defaultdict(lambda:[0,0,0])
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    for i,batch in enumerate(loader):
                        torch.manual_seed(image_seed(seed,batch[0]["image_id"]))
                        torch.cuda.synchronize()
                        tick=time.perf_counter()
                        noise=proposal_noise(1,count,300,model.device)
                        prediction=model(batch,inference_noise=noise)
                        torch.cuda.synchronize()
                        ms=1000*(time.perf_counter()-tick)
                        if 20<=i<220:
                            latencies.append(ms)
                        evaluator.process(batch,prediction)
                        coords=holder.pop("coords").cpu().numpy()[:,0]
                        if coords.shape[:2]!=(4,count):
                            raise RuntimeError("Actual stage execution does not match requested P/H")
                        accumulate_coverage(coverage,coords,records[i],batch[0]["image"].shape[-2:])
                metrics=evaluator.evaluate(img_ids=manifest["image_ids"])["bbox"]
                row=dict(train_seed=train_seed,eval_seed=seed,proposals=count,images=len(records),
                         full_dataset=len(records)==len(full_records),**metrics,
                         latency_p50_ms=float(np.median(latencies)),latency_p95_ms=float(np.percentile(latencies,95)),
                         measured_images=len(latencies),peak_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
                rows.append(row)
                (arm_dir/"metrics.json").write_text(json.dumps(row,indent=2)+"\n")
                for (stage,group),(total,h50,h75) in sorted(coverage.items()):
                    coverage_rows.append(dict(train_seed=train_seed,eval_seed=seed,proposals=count,stage=stage,group=group,
                                              objects=total,hits50=h50,hits75=h75,coverage50=h50/total,coverage75=h75/total))
                write_rows(args.out_dir/"metrics.csv",rows)
                write_rows(args.out_dir/"coverage.csv",coverage_rows)
                print("[p1] "+json.dumps(row),flush=True)
        hook.remove()
        del model,reference,actual,noise,prediction
        torch.cuda.empty_cache()
    (args.out_dir/"COMPLETE.json").write_text(json.dumps(dict(evaluations=len(rows),checkpoints=len(args.train_seeds)))+"\n")


if __name__=="__main__":
    main()
