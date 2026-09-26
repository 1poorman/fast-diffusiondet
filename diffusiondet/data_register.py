# -*- coding: utf-8 -*-
"""
数据集注册 (M0)。

两个数据集的 COCO json 都存在同一个陷阱：``categories`` 里含 ``(0, '_background_')``，
但 ``annotations`` 的 ``category_id`` 从不等于 0。若直接用 detectron2 的默认映射，
会多出一个永远没有 GT 的空类，污染 Focal loss 的先验与 mAP 分母。

因此这里显式构造 ``id_map = {cid: index}``（**排除 id == 0**），
并把 ``category_id`` 直接映射成 contiguous id，避免依赖 detectron2 内部行为。

数据集根目录解析顺序：
    1. ``register_all(root=...)`` 的显式入参
    2. ``cfg.DATASET_ROOT``（可由命令行 ``--opts DATASET_ROOT <path>`` 覆盖）
    3. 环境变量 ``FASTDD_DATASET``
    4. 仓库同级目录 ``<repo>/../dataset``
"""

import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode

__all__ = ["DATASET_ROOT_DEFAULT", "PREDEFINED_SPLITS", "resolve_dataset_root", "register_all"]

# 仓库根目录的同级 dataset 目录：<repo>/../dataset
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT_DEFAULT = os.path.normpath(
    os.path.join(_REPO_ROOT, os.pardir, "dataset")
)

PREDEFINED_SPLITS = {
    # name: (image_root_rel, json_rel, num_thing_classes)
    "pls_train": ("PLS/Plantv2/train2017", "PLS/Plantv2/annotations/instances_train2017.json", 16),
    "pls_val": ("PLS/Plantv2/val2017", "PLS/Plantv2/annotations/instances_val2017.json", 16),
    "pls_test": ("PLS/Plantv2/test2017", "PLS/Plantv2/annotations/instances_test2017.json", 16),
    "sdd_train": ("SDD/Strawberry/train2017", "SDD/Strawberry/annotations/instances_train2017.json", 7),
    "sdd_val": ("SDD/Strawberry/val2017", "SDD/Strawberry/annotations/instances_val2017.json", 7),
    "wheat_train": ("WHEAT/strat_real/train2017", "WHEAT/wheat_seg_strat/annotations/instances_train2017.json", 12),
    "wheat_val": ("WHEAT/strat_real/val2017", "WHEAT/wheat_seg_strat/annotations/instances_val2017.json", 12),
}

_BACKGROUND_IDS = (0,)
_BACKGROUND_NAMES = ("_background_",)


def resolve_dataset_root(root=None, cfg=None):
    """按「显式入参 > cfg > 环境变量 > 默认路径」解析数据集根目录。"""
    candidates = [
        root,
        getattr(cfg, "DATASET_ROOT", "") if cfg is not None else "",
        os.environ.get("FASTDD_DATASET", ""),
        DATASET_ROOT_DEFAULT,
    ]
    for c in candidates:
        if c and os.path.isdir(c):
            return os.path.normpath(c)
    raise FileNotFoundError(
        "找不到数据集根目录。已尝试:\n  " + "\n  ".join(repr(c) for c in candidates if c)
        + "\n请设置环境变量 FASTDD_DATASET 或命令行传入 DATASET_ROOT。"
    )


_CACHE_DIRNAME = ".cache_fastdd"


def make_eval_json(json_file, name, excluded_cat_ids, root):
    """
    COCOEvaluator 是直接用 pycocotools 从 json 重建 category 列表的，
    而我们的源 json 里含 id=0 的 '_background_'，会导致
    ``assert len(class_names) == precisions.shape[2]`` 失败。

    因此为评测 split 生成一份剔除 background 的干净 json（带缓存，只写一次）。
    注意：**保留原有 category id**（1..K），因为 COCO 允许非连续 id，
    而 thing_dataset_id_to_contiguous_id 负责 1..K -> 0..K-1 的映射。
    """
    cache_dir = os.path.join(root, _CACHE_DIRNAME)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, "{}.clean.json".format(name))
    if os.path.isfile(cache_file) and os.path.getmtime(cache_file) >= os.path.getmtime(json_file):
        return cache_file

    with open(json_file, "r", encoding="utf-8") as f:
        j = json.load(f)

    cats = [c for c in j.get("categories", []) if c["id"] not in excluded_cat_ids
            and c["name"] not in _BACKGROUND_NAMES]
    keep_ids = {c["id"] for c in cats}
    anns = [a for a in j.get("annotations", []) if a["category_id"] in keep_ids]

    clean = {
        "info": j.get("info", {}),
        "licenses": j.get("licenses", []),
        "images": j.get("images", []),
        "categories": cats,
        "annotations": anns,
    }
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(clean, f)
    print("[data_register]   生成评测用干净 json -> {} (cats={}, anns={}/{})".format(
        os.path.relpath(cache_file, root), len(cats), len(anns), len(j.get("annotations", []))))
    return cache_file


def _load_and_convert(json_file, image_root):
    """读取一个 COCO json，返回 (dataset_dicts, thing_classes, id_map, (n_bg, n_crowd))。"""
    from collections import OrderedDict

    with open(json_file, "r", encoding="utf-8") as f:
        j = json.load(f)

    # ---------- 类别：剔除 background 后按 id 升序建立 contiguous 映射 ----------
    cats = [c for c in j.get("categories", [])
            if c["id"] not in _BACKGROUND_IDS and c["name"] not in _BACKGROUND_NAMES]
    cats = sorted(cats, key=lambda c: c["id"])
    thing_classes = [c["name"] for c in cats]
    id_map = {c["id"]: i for i, c in enumerate(cats)}

    # ---------- 图片与标注 ----------
    images = {im["id"]: im for im in j.get("images", [])}
    # 一张图可能有多条标注 -> 聚合
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    skipped_bg = skipped_crowd = 0
    for a in j.get("annotations", []):
        if a["category_id"] not in id_map:
            skipped_bg += 1
            continue
        if a.get("iscrowd", 0) != 0:
            skipped_crowd += 1
            continue
        anns_by_img[a["image_id"]].append(a)

    dataset_dicts = []
    for img_id in sorted(images.keys()):
        im = images[img_id]
        record = OrderedDict()
        record["file_name"] = os.path.join(image_root, im["file_name"])
        record["image_id"] = img_id
        record["height"] = im["height"]
        record["width"] = im["width"]
        objs = []
        for a in anns_by_img.get(img_id, []):
            obj = OrderedDict()
            obj["bbox"] = a["bbox"]
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = id_map[a["category_id"]]
            if "segmentation" in a:
                obj["segmentation"] = a["segmentation"]
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts, thing_classes, id_map, (skipped_bg, skipped_crowd)


def register_all(root=None, cfg=None, dataset_names=None, verbose=True):
    """注册 PLS / SDD 全部 split。重复调用幂等（已注册则跳过）。"""
    root = resolve_dataset_root(root, cfg)
    names = dataset_names or list(PREDEFINED_SPLITS.keys())

    registered = []
    for name in names:
        if name in DatasetCatalog.list():
            continue
        image_root_rel, json_rel, _ = PREDEFINED_SPLITS[name]
        image_root = os.path.join(root, *image_root_rel.split("/"))
        json_file = os.path.join(root, *json_rel.split("/"))
        if not os.path.isfile(json_file):
            raise FileNotFoundError("标注文件不存在: {}".format(json_file))
        if not os.path.isdir(image_root):
            raise FileNotFoundError("图片目录不存在: {}".format(image_root))

        dicts, thing_classes, id_map, (n_bg, n_crowd) = _load_and_convert(json_file, image_root)

        # 评测 split 必须用剔除 background 的干净 json，否则 COCOEvaluator 会因类别数不一致 assert 失败
        md_json_file = json_file
        if not name.endswith("_train"):
            md_json_file = make_eval_json(json_file, name, _BACKGROUND_IDS, root)

        DatasetCatalog.register(name, lambda d=dicts: d)
        MetadataCatalog.get(name).set(
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=id_map,
            json_file=md_json_file,
            image_root=image_root,
            evaluator_type="coco",
        )
        registered.append((name, len(dicts), len(thing_classes)))
        if verbose:
            print("[data_register] {:<12} images={:<6} classes={:<3} 跳过 background/crowd 标注={}/{}".format(
                name, len(dicts), len(thing_classes), n_bg, n_crowd))

    if verbose:
        print("[data_register] root = {}".format(root))
    return registered


if __name__ == "__main__":
    register_all()
