# tests/memory/test_tile_config.py


def test_tiled_config_counts(tiled_config):
    assert tiled_config.uses_tiling is True
    assert tiled_config.effective_tile_size == (174, 174)
    assert tiled_config.effective_stride == (87, 87)
    assert tiled_config.n_tiles_h == 3
    assert tiled_config.n_tiles_w == 3
    assert tiled_config.tiles_per_image == 9
    assert tiled_config.effective_tile_batch == 288


def test_no_tiling_config_counts(no_tiling_config):
    assert no_tiling_config.uses_tiling is False
    assert no_tiling_config.effective_tile_size == (348, 348)
    assert no_tiling_config.effective_stride == (348, 348)
    assert no_tiling_config.n_tiles_h == 1
    assert no_tiling_config.n_tiles_w == 1
    assert no_tiling_config.tiles_per_image == 1
    assert no_tiling_config.effective_tile_batch == 32
