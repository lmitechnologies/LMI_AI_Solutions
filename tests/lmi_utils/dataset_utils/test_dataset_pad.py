import json

from lmi_utils.dataset_utils.ops.dataset_pad import clip_shapes
from lmi_utils.dataset_utils.representations import Polygon, PolygonAnnotation


def test_a_clipped_polygon_keeps_json_serializable_points():
    polygon = PolygonAnnotation("p", "dent", Polygon(points=[[10, 10], [150, 10], [150, 50]]))
    clip_shapes([polygon], 100, 100)
    assert json.loads(polygon.to_json())["value"]["points"] == [[10.0, 10.0], [100.0, 10.0], [100.0, 50.0]]
