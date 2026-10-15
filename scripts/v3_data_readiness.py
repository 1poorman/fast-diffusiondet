"""D1: read-only image/split audit before cross-dataset model confirmation."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import itertools
import json
import math
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diffusiondet.data_register import PREDEFINED_SPLITS, resolve_image_file
from scripts.v3_summarize_distill import write_rows

DOMAINS = {'pls': ('pls_train', 'pls_val', 'pls_test'),
           'sdd': ('sdd_train', 'sdd_val'), 'wheat': ('wheat_train', 'wheat_val')}
CHECKPOINTS = {'pls': 'output/lab/pls.res50.bs8/model_final.pth',
               'wheat': 'output/lab/wheat.res50.bs8/model_final.pth'}


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def audit_split(name, root):
    from PIL import Image
    image_rel, annotation_rel, expected_classes = PREDEFINED_SPLITS[name]
    annotation, image_root = root / annotation_rel, root / image_rel
    source = json.loads(annotation.read_text())
    categories = {c['id']: c['name'] for c in source['categories']
                  if c['id'] != 0 and c['name'] != '_background_'}
    images, annotations = source['images'], source['annotations']
    image_ids, ann_ids = [r['id'] for r in images], [r['id'] for r in annotations]
    errors = []
    if len(set(image_ids)) != len(image_ids):
        errors.append('duplicate_image_ids')
    if len(set(ann_ids)) != len(ann_ids):
        errors.append('duplicate_annotation_ids')
    if len({c['id'] for c in source['categories']}) != len(source['categories']):
        errors.append('duplicate_category_ids')
    if len(categories) != expected_classes:
        errors.append('unexpected_class_count')
    valid_ids = set(image_ids)
    annotated = defaultdict(set)
    counts = Counter()
    invalid_boxes, orphan_annotations, unknown_categories = [], [], []
    for a in annotations:
        if a['image_id'] not in valid_ids:
            orphan_annotations.append(a['id'])
        if a['category_id'] not in {c['id'] for c in source['categories']}:
            unknown_categories.append(a['id'])
        if a['category_id'] not in categories or a.get('iscrowd', 0):
            continue
        box = a.get('bbox', [])
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or min(box[2:]) <= 0:
            invalid_boxes.append(a['id'])
        counts[a['category_id']] += 1
        annotated[a['category_id']].add(a['image_id'])
    for key, value in [('orphan_annotations', orphan_annotations), ('unknown_categories', unknown_categories),
                       ('invalid_boxes', invalid_boxes)]:
        if value:
            errors.append(key)
    records = []
    for item in images:
        original = image_root / item['file_name']
        resolved = Path(resolve_image_file(image_root, item['file_name']))
        row = dict(image_id=item['id'], file_name=item['file_name'], resolved_path=str(resolved),
                   original_exists=original.is_file(), path_changed=str(original) != str(resolved),
                   exists=resolved.is_file(), readable=False, dimensions_match=False)
        if row['exists']:
            try:
                data = resolved.read_bytes()
                with Image.open(BytesIO(data)) as image:
                    rgb = image.convert('RGB')
                    row.update(file_sha256=sha256(data).hexdigest(),
                               rgb_sha256=sha256(f'{rgb.width},{rgb.height}:'.encode() + rgb.tobytes()).hexdigest(),
                               actual_width=rgb.width, actual_height=rgb.height, readable=True,
                               dimensions_match=rgb.size == (item['width'], item['height']))
            except (OSError, ValueError) as error:
                row['error'] = str(error)
        records.append(row)
    classes = [dict(split=name, category_id=cid, category=label, annotations=counts[cid], images=len(annotated[cid]))
               for cid, label in sorted(categories.items())]
    summary = dict(split=name, images=len(images), annotations=len(annotations),
                   effective_annotations=sum(counts.values()), declared_classes=len(categories),
                   observed_classes=sum(n > 0 for n in counts.values()),
                   original_missing=sum(not r['original_exists'] for r in records),
                   changed_paths=sum(r['path_changed'] for r in records),
                   missing=sum(not r['exists'] for r in records), unreadable=sum(not r['readable'] for r in records),
                   dimension_mismatches=sum(r['readable'] and not r['dimensions_match'] for r in records),
                   structural_errors=errors)
    detail = dict(annotation_path=str(annotation), annotation_sha256=digest(annotation), categories=categories,
                  invalid_box_ids=invalid_boxes, orphan_annotation_ids=orphan_annotations,
                  unknown_category_annotation_ids=unknown_categories, records=records)
    return summary, classes, detail


def compare_splits(left, right, left_records, right_records):
    """Count exact duplicate groups and participating images, not just pair products."""
    rows = []
    for field in ('file_sha256', 'rgb_sha256'):
        groups = []
        a, b = defaultdict(list), defaultdict(list)
        for record in left_records:
            if field in record:
                a[record[field]].append(record['image_id'])
        for record in right_records:
            if field in record:
                b[record[field]].append(record['image_id'])
        for value in sorted(a.keys() & b.keys()):
            groups.append(dict(sha256=value, left_image_ids=sorted(a[value]), right_image_ids=sorted(b[value])))
        rows.append(dict(left=left, right=right, hash_type=field, duplicate_groups=len(groups),
                         left_images=sum(len(g['left_image_ids']) for g in groups),
                         right_images=sum(len(g['right_image_ids']) for g in groups), groups=groups))
    return rows


def confirmation_readiness(domain, names, summaries, classes, details, overlaps, checkpoint_available):
    train = next(n for n in names if n.endswith('_train'))
    trained = {r['category_id'] for r in classes if r['split'] == train and r['annotations'] > 0}
    result = []
    for name in names:
        if name == train:
            continue
        unseen = [r for r in classes if r['split'] == name and r['annotations'] > 0 and r['category_id'] not in trained]
        relevant = [r for r in overlaps if name in (r['left'], r['right'])]
        split = next(r for r in summaries if r['split'] == name)
        training = next(r for r in summaries if r['split'] == train)
        checks = dict(images_readable=all(r['unreadable'] == 0 and r['dimension_mismatches'] == 0 for r in (split, training)),
                      annotation_structure_valid=not split['structural_errors'] and not training['structural_errors'],
                      category_mapping_matches=details[name]['categories'] == details[train]['categories'],
                      no_exact_cross_split_overlap=all(r['duplicate_groups'] == 0 for r in relevant),
                      all_evaluation_classes_seen_in_training=not unseen, checkpoint_available=checkpoint_available)
        result.append(dict(domain=domain, split=name, checks=checks, closed_set_execution_ready=all(checks.values()),
                           unseen_classes=unseen,
                           prior_test_usage='unknown' if name.endswith('_test') else 'development set; used historically',
                           independent_confirmation_completed=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root', type=Path, default=Path('datasets'))
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--prefix', default='v3_data_readiness')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    sources = ['scripts/v3_data_readiness.py', 'diffusiondet/data_register.py', 'docs/v3/DATA_READINESS.md',
               'scripts/v3_summarize_distill.py']
    for name in sources:
        target = args.out_dir / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(name, target)
    summaries, classes, details, overlaps, readiness = [], [], {}, [], []
    for domain, names in DOMAINS.items():
        for name in names:
            summary, coverage, detail = audit_split(name, args.dataset_root)
            summaries.append(summary)
            classes.extend(coverage)
            details[name] = detail
            (args.out_dir / (name + '.json')).write_text(json.dumps(detail, indent=2) + '\n')
            print('[d1-split] ' + json.dumps(summary), flush=True)
        for left, right in itertools.combinations(names, 2):
            overlaps.extend(compare_splits(left, right, details[left]['records'], details[right]['records']))
        checkpoint = CHECKPOINTS.get(domain, 'runs/v3/time_alignment_20261011/seed10_uniform/model_final.pth')
        readiness.extend(confirmation_readiness(domain, names, summaries, classes, details, overlaps, Path(checkpoint).is_file()))
    for name, detail in details.items():
        if digest(detail['annotation_path']) != detail['annotation_sha256']:
            raise RuntimeError('Source annotation changed during audit: ' + name)
    artifacts = {name + '.json': digest(args.out_dir / (name + '.json')) for name in details}
    import PIL
    report = dict(completed=True, images=sum(r['images'] for r in summaries), run_dir=str(args.out_dir),
                  dataset_root=str(args.dataset_root.resolve()), python=sys.version, pillow=PIL.__version__,
                  source_sha256={name: digest(name) for name in sources},
                  annotation_sha256={name: d['annotation_sha256'] for name, d in details.items()},
                  artifacts_sha256=artifacts, readiness=readiness,
                  checkpoints={domain: dict(path=name, available=Path(name).is_file()) for domain, name in CHECKPOINTS.items()},
                  limitations=['Exact bytes and decoded RGB only; near-duplicate similarity and acquisition groups not established.',
                               'Test usage history cannot be established from absent local logs.',
                               'No model AP, speed or generalization claim from this data audit.'])
    serial = [dict(r, structural_errors=';'.join(r['structural_errors'])) for r in summaries]
    write_rows(args.out_dir / 'splits.csv', serial)
    write_rows(args.out_dir / 'classes.csv', classes)
    (args.out_dir / 'overlap.json').write_text(json.dumps(overlaps, indent=2) + '\n')
    (args.out_dir / 'COMPLETE.json').write_text(json.dumps(report, indent=2) + '\n')
    raw = Path('results/raw')
    raw.mkdir(exist_ok=True, parents=True)
    for name in ('splits.csv', 'classes.csv', 'overlap.json', 'COMPLETE.json'):
        shutil.copyfile(args.out_dir / name, raw / (args.prefix + '_' + name))
    print('[d1-complete] ' + json.dumps(dict(images=report['images'], readiness=readiness)), flush=True)


if __name__ == '__main__':
    main()
