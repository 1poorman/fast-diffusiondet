import json

from PIL import Image, PngImagePlugin

from scripts.v3_data_readiness import audit_split, compare_splits, confirmation_readiness


def test_exact_duplicate_counts_include_all_participating_images():
    left = [dict(image_id=i, file_sha256='same', rgb_sha256='pixels') for i in (1, 2)]
    right = [dict(image_id=4, file_sha256='reencoded', rgb_sha256='pixels'),
             dict(image_id=5, file_sha256='same', rgb_sha256='pixels')]
    files, pixels = compare_splits('train', 'test', left, right)
    assert files['duplicate_groups'] == 1 and files['left_images'] == 2 and files['right_images'] == 1
    assert pixels['duplicate_groups'] == 1 and pixels['left_images'] == 2 and pixels['right_images'] == 2
    assert pixels['groups'][0]['left_image_ids'] == [1, 2]


def test_decoded_pixel_audit_finds_reencoded_images_and_metadata_errors(tmp_path, monkeypatch):
    import scripts.v3_data_readiness as implementation
    monkeypatch.setitem(implementation.PREDEFINED_SPLITS, 'fixture', ('test2017', 'annotations.json', 2))
    root = tmp_path / 'test2017'
    root.mkdir()
    image = Image.new('RGB', (4, 3), (10, 20, 30))
    image.save(root / 'first.png')
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('note', 'different bytes, same RGB pixels')
    image.save(root / 'second.png', pnginfo=metadata)
    source = dict(images=[dict(id=1, file_name=r'test2017\first.png', width=4, height=3),
                          dict(id=2, file_name='second.png', width=5, height=3),
                          dict(id=3, file_name='missing.png', width=4, height=3)],
                  categories=[dict(id=1, name='seen'), dict(id=12, name='unseen')],
                  annotations=[dict(id=8, image_id=1, category_id=1, bbox=[0, 0, 2, 2]),
                               dict(id=9, image_id=99, category_id=12, bbox=[0, 0, -1, 2])])
    (tmp_path / 'annotations.json').write_text(json.dumps(source))
    summary, classes, detail = audit_split('fixture', tmp_path)
    assert summary['original_missing'] == 2 and summary['changed_paths'] == 1
    assert summary['missing'] == 1 and summary['dimension_mismatches'] == 1
    assert set(summary['structural_errors']) == {'orphan_annotations', 'invalid_boxes'}
    first, second = detail['records'][:2]
    assert first['file_sha256'] != second['file_sha256'] and first['rgb_sha256'] == second['rgb_sha256']
    assert detail['orphan_annotation_ids'] == [9] and detail['invalid_box_ids'] == [9]


def test_closed_set_readiness_requires_training_coverage_and_domain_weights():
    names = ('example_train', 'example_test')
    summaries = [dict(split=n, unreadable=0, dimension_mismatches=0, structural_errors=[]) for n in names]
    details = {n: dict(categories={1: 'seen', 12: 'unseen'}) for n in names}
    classes = [dict(split='example_train', category_id=1, category='seen', annotations=10),
               dict(split='example_test', category_id=12, category='unseen', annotations=2)]
    result = confirmation_readiness('example', names, summaries, classes, details, [], False)[0]
    assert not result['closed_set_execution_ready']
    assert not result['checks']['all_evaluation_classes_seen_in_training']
    assert not result['checks']['checkpoint_available']
    assert result['unseen_classes'] == [classes[1]]
    assert result['prior_test_usage'] == 'unknown'
    assert result['independent_confirmation_completed'] is False
    classes.append(dict(split='example_train', category_id=12, category='unseen', annotations=3))
    ready = confirmation_readiness('example', names, summaries, classes, details, [], True)[0]
    assert ready['closed_set_execution_ready']
    duplicates = [dict(left='example_train', right='example_test', duplicate_groups=1)]
    assert not confirmation_readiness('example', names, summaries, classes, details, duplicates, True)[0]['closed_set_execution_ready']
    details['example_test']['categories'] = {1: 'different', 12: 'unseen'}
    assert not confirmation_readiness('example', names, summaries, classes, details, [], True)[0]['checks']['category_mapping_matches']


def test_sensitivity_excludes_only_sdd_validation_ids_once():
    from scripts.v3_data_sensitivity import exact_duplicate_ids
    rows = [dict(left='sdd_train', right='sdd_val', groups=[dict(left_image_ids=[3], right_image_ids=[7, 8])]),
            dict(left='sdd_train', right='sdd_val', groups=[dict(left_image_ids=[3], right_image_ids=[7, 8])]),
            dict(left='pls_train', right='pls_test', groups=[dict(right_image_ids=[99])])]
    assert exact_duplicate_ids(rows) == [7, 8]


def test_sensitivity_requires_finite_reproducible_full_validation_metrics():
    import pytest
    from scripts.v3_data_sensitivity import verify_replay
    assert verify_replay({'AP': 67.}, {'AP': '67.0'}) == 0
    for changed in (67.00001, float('nan')):
        with pytest.raises(ValueError):
            verify_replay({'AP': changed}, {'AP': '67.0'})
