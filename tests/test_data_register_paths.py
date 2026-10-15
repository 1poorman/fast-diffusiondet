import json
import os

import pytest

from diffusiondet.data_register import _load_and_convert, resolve_image_file


@pytest.mark.parametrize('exported', ['image.jpg', 'test2017/image.jpg', r'test2017\image.jpg'])
def test_split_prefix_and_windows_separator_resolve_same_image(tmp_path, exported):
    root = tmp_path / 'test2017'
    root.mkdir()
    expected = root / 'image.jpg'
    expected.write_bytes(b'image')
    assert resolve_image_file(root, exported) == str(expected)


def test_nested_paths_are_preserved_instead_of_flattened(tmp_path):
    root = tmp_path / 'test2017'
    nested = root / 'class_name'
    nested.mkdir(parents=True)
    expected = nested / 'image.jpg'
    expected.write_bytes(b'nested')
    (root / 'image.jpg').write_bytes(b'different image')
    assert resolve_image_file(root, r'class_name\image.jpg') == str(expected)
    assert resolve_image_file(root, r'unknown\image.jpg') == str(root / r'unknown\image.jpg')


def test_existing_literal_path_takes_precedence(tmp_path):
    root = tmp_path / 'test2017'
    nested = root / 'test2017'
    nested.mkdir(parents=True)
    expected = nested / 'image.jpg'
    expected.write_bytes(b'literal')
    (root / 'image.jpg').write_bytes(b'different image')
    assert resolve_image_file(root, 'test2017/image.jpg') == str(expected)
    if os.name != 'nt':
        literal = root / r'test2017\image.jpg'
        literal.write_bytes(b'literal backslash')
        assert resolve_image_file(root, r'test2017\image.jpg') == str(literal)


def test_missing_path_is_not_redirected_into_another_split(tmp_path):
    root = tmp_path / 'test2017'
    root.mkdir()
    sibling = tmp_path / 'train2017'
    sibling.mkdir()
    (sibling / 'image.jpg').write_bytes(b'training image')
    assert resolve_image_file(root, r'test2017\image.jpg') == str(root / r'test2017\image.jpg')


def test_loader_fixes_path_without_rewriting_annotations_or_category_ids(tmp_path):
    root = tmp_path / 'test2017'
    root.mkdir()
    (root / 'image.jpg').write_bytes(b'image')
    source = tmp_path / 'annotations.json'
    data = dict(images=[dict(id=7, file_name=r'test2017\image.jpg', height=10, width=20)],
                categories=[dict(id=0, name='_background_'), dict(id=4, name='seen'), dict(id=12, name='unseen')],
                annotations=[dict(id=0, image_id=7, category_id=12, bbox=[1, 2, 3, 4], iscrowd=0)])
    source.write_text(json.dumps(data))
    before = source.read_bytes()
    records, classes, mapping, skipped = _load_and_convert(source, str(root))
    assert records[0]['file_name'] == str(root / 'image.jpg')
    assert records[0]['annotations'][0]['category_id'] == 1
    assert classes == ['seen', 'unseen'] and mapping == {4: 0, 12: 1} and skipped == (0, 0)
    assert source.read_bytes() == before
