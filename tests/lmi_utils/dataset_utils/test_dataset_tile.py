import numpy as np
import pytest

from lmi_utils.dataset_utils.ops.dataset_tile import tile_annotated_image, tile_dataset
from lmi_utils.dataset_utils.representations import (
    Box,
    BoxAnnotation,
    Dataset,
    FileAnnotations,
    KeypointAnnotation,
    Label,
    Mask,
    MaskAnnotation,
    Point2d,
    Polygon,
    PolygonAnnotation,
)
from lmi_utils.image_utils.tiler import Tiler


def _image(h=100, w=100):
    return np.arange(h * w * 3, dtype=np.uint8).reshape(h, w, 3)


def _box(id_, x1, y1, x2, y2, label="defect"):
    return BoxAnnotation(id_, label, Box(x1, y1, x2, y2))


def test_grid_matches_the_inference_tiler():
    """The dataset op and predict() must cut the same grid, or truncation flags line up with the wrong edges."""
    import torch

    image = _image(100, 150)
    tiles = tile_annotated_image(image, [], tile_size=64, stride=32)

    tiler = Tiler([64, 64], [32, 32])
    expected = tiler.tile(torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0))

    assert len(tiles) == expected.shape[0]
    assert [(r, c) for r, c, _, _ in tiles] == [(i // tiler.n_tiles[1], i % tiler.n_tiles[1]) for i in range(len(tiles))]
    for i, (_, _, tile_im, _) in enumerate(tiles):
        assert np.array_equal(tile_im, expected[i].permute(1, 2, 0).numpy())


def test_boxes_are_shifted_and_clipped_into_each_tile():
    # a box spanning the seam at x=50 between tile 0 (x 0..64) and tile 1 (x 50..114)
    annotations = [_box("b", 40, 10, 70, 30)]
    tiles = tile_annotated_image(_image(64, 114), annotations, tile_size=64, stride=50)

    by_col = {col: annots for _, col, _, annots in tiles}
    assert len(by_col) == 2

    left = by_col[0][0].value
    assert (left.x_min, left.y_min, left.x_max, left.y_max) == (40, 10, 64, 30), "clipped at the tile's right edge"

    right = by_col[1][0].value
    assert (right.x_min, right.y_min, right.x_max, right.y_max) == (0, 10, 20, 30), "shifted by the tile origin"


def test_a_box_outside_a_tile_is_dropped():
    tiles = tile_annotated_image(_image(64, 114), [_box("b", 0, 0, 20, 20)], tile_size=64, stride=50)
    counts = {col: len(annots) for _, col, _, annots in tiles}
    assert counts == {0: 1, 1: 0}


def test_interior_fragments_survive():
    """An object wider than a tile leaves fragments showing none of its edges; those must be kept."""
    annotations = [_box("wide", 0, 0, 200, 60)]
    tiles = tile_annotated_image(_image(64, 200), annotations, tile_size=64, stride=64)

    assert all(len(annots) == 1 for _, _, _, annots in tiles)
    middle = tiles[1][3][0].value
    assert (middle.x_min, middle.x_max) == (0, 64), "the middle tile sees a full-width slab, not an edge"


def test_min_label_size_drops_slivers_only():
    annotations = [_box("sliver", 63, 10, 70, 30)]  # 1 px inside tile 0, 6 px inside tile 1
    tiles = tile_annotated_image(_image(64, 114), annotations, tile_size=64, stride=50, min_label_size=3.0)

    counts = {col: len(annots) for _, col, _, annots in tiles}
    assert counts == {0: 0, 1: 1}


def test_polygon_is_clipped_by_intersection_not_vertex_clamping():
    poly = PolygonAnnotation("p", "defect", Polygon([[40, 10], [80, 10], [80, 30], [40, 30]]))
    tiles = tile_annotated_image(_image(64, 114), [poly], tile_size=64, stride=50)

    left = np.array(tiles[0][3][0].value.points)
    assert left[:, 0].max() == pytest.approx(64.0)
    assert left[:, 1].min() == pytest.approx(10.0) and left[:, 1].max() == pytest.approx(30.0)


def test_a_polygon_split_into_pieces_becomes_several_labels():
    # a U rotated to open rightwards: the tile edge at x=64 cuts it into two disjoint arms
    u = PolygonAnnotation("u", "defect", Polygon([[10, 10], [90, 10], [90, 20], [30, 20], [30, 40], [90, 40], [90, 50], [10, 50]]))
    tiles = tile_annotated_image(_image(64, 114), [u], tile_size=64, stride=50)

    right = tiles[1][3]
    assert len(right) == 2, "the two arms reaching into the right tile are separate labels"
    assert {a.id for a in right} == {"u_p0", "u_p1"}
    assert all(a.label_id == "defect" for a in right)


def test_masks_are_sliced_per_tile():
    mask = np.zeros((64, 114), np.uint8)
    mask[10:30, 40:70] = 1
    tiles = tile_annotated_image(_image(64, 114), [MaskAnnotation("m", "defect", Mask(mask))], tile_size=64, stride=50)

    left = tiles[0][3][0].value.to_numpy(h=64, w=64)
    assert left.shape == (64, 64)
    assert left[10:30, 40:64].all() and left.sum() == 20 * 24

    right = tiles[1][3][0].value.to_numpy(h=64, w=64)
    assert right[10:30, 0:20].all() and right.sum() == 20 * 20


def test_padding_is_bottom_right_only():
    tiles = tile_annotated_image(np.full((50, 50, 3), 7, np.uint8), [], tile_size=64, stride=64)

    assert len(tiles) == 1
    tile = tiles[0][2]
    assert tile.shape == (64, 64, 3)
    assert (tile[:50, :50] == 7).all()
    assert (tile[50:, :] == 0).all() and (tile[:, 50:] == 0).all()


def test_keypoints_and_oriented_boxes_are_rejected():
    with pytest.raises(ValueError, match="does not support keypoints"):
        tile_annotated_image(_image(), [KeypointAnnotation("k", "defect", Point2d(10, 10))], tile_size=64, stride=64)

    obb = BoxAnnotation("b", "defect", Box(10, 10, 20, 20, angle=30))
    with pytest.raises(ValueError, match="does not support oriented boxes"):
        tile_annotated_image(_image(), [obb], tile_size=64, stride=64)


def test_source_annotations_are_not_mutated():
    annotations = [_box("b", 40, 10, 70, 30)]
    tile_annotated_image(_image(64, 114), annotations, tile_size=64, stride=50)

    assert annotations[0].value.to_numpy()[:4].tolist() == [40, 10, 70, 30]


def test_tile_dataset_expands_files_and_records_the_source():
    dataset = Dataset(
        labels=[Label(id="defect")],
        files=[FileAnnotations(id="1", path="a.png", height=64, width=114, annotations=[_box("b", 40, 10, 70, 30)])],
    )
    images = {"a.png": _image(64, 114)}

    tiled_images, tiled = tile_dataset(dataset, images, tile_size=64, stride=50)

    assert [f.path for f in tiled.files] == ["id1_r0_c0_a.png", "id1_r0_c1_a.png"]
    assert [f.id for f in tiled.files] == ["1_r0_c0", "1_r0_c1"]
    assert all(f.source_id == "1" for f in tiled.files), "tiles of one image must stay on one side of a split"
    assert all((f.height, f.width) == (64, 64) for f in tiled.files)
    assert set(tiled_images) == {f.path for f in tiled.files}


def test_tile_dataset_keeps_empty_tiles_for_the_caller_to_drop():
    dataset = Dataset(
        labels=[Label(id="defect")],
        files=[FileAnnotations(id="1", path="a.png", height=64, width=114, annotations=[_box("b", 0, 0, 20, 20)])],
    )

    _, tiled = tile_dataset(dataset, {"a.png": _image(64, 114)}, tile_size=64, stride=50)
    assert len(tiled.files) == 2

    tiled.delete_empty_files()
    assert [f.path for f in tiled.files] == ["id1_r0_c0_a.png"]


def test_tile_dataset_keeps_several_source_images_apart():
    dataset = Dataset(
        labels=[Label(id="defect")],
        files=[
            FileAnnotations(id="1", path="sub_a/im.png", height=64, width=114, annotations=[_box("b", 40, 10, 70, 30)]),
            FileAnnotations(id="2", path="sub_b/im.png", height=64, width=114, annotations=[_box("b", 40, 10, 70, 30)]),
        ],
    )
    images = {"sub_a/im.png": _image(64, 114), "sub_b/im.png": _image(64, 114)}

    tiled_images, tiled = tile_dataset(dataset, images, tile_size=64, stride=50)

    assert len(tiled.files) == 4
    assert len(tiled_images) == 4, "same basename in two folders must not collide into one output image"
    assert [f.source_id for f in tiled.files] == ["1", "1", "2", "2"]
    assert [f.path for f in tiled.files] == ["id1_r0_c0_im.png", "id1_r0_c1_im.png", "id2_r0_c0_im.png", "id2_r0_c1_im.png"]


def test_non_square_tiles_and_strides():
    annotations = [_box("b", 10, 10, 120, 40)]
    tiles = tile_annotated_image(_image(60, 140), annotations, tile_size=[32, 64], stride=[28, 50])

    # h: 32 + ceil((60-32)/28)*28 = 60 -> 2 rows; w: 64 + ceil((140-64)/50)*50 = 164 -> 3 cols
    assert [(r, c) for r, c, _, _ in tiles] == [(r, c) for r in range(2) for c in range(3)]
    assert all(tile.shape[:2] == (32, 64) for _, _, tile, _ in tiles)

    top_left = tiles[0][3][0].value
    assert (top_left.x_min, top_left.y_min, top_left.x_max, top_left.y_max) == (10, 10, 64, 32)
