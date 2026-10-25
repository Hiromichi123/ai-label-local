from pathlib import Path
from PIL import Image
import pytest
from app.image_ops import square_transform
from app.labels import parse_yolo, serialize
from app.schemas import Box


def test_center_crop_maps_box():
    transform = square_transform(400, 200, "center_crop")
    assert transform.box((100, 50, 300, 150)) == (0, 100, 400, 300)


def test_letterbox_keeps_square_output():
    transform = square_transform(400, 200, "letterbox_pad")
    assert transform.output_size == 400
    assert transform.box((0, 0, 400, 200)) == (0, 0, 400, 200)


def test_yolo_round_trip():
    boxes = parse_yolo("1 0.5 0.5 0.25 0.5\n", 400, 200, 2)
    assert boxes[0].class_id == 1
    assert serialize(boxes, 400, 200).startswith("1 0.500000 0.500000")


def test_yolo_rejects_invalid_values():
    with pytest.raises(ValueError):
        parse_yolo("0 0.5 0.5 1.2 0.2", 100, 100, 1)
