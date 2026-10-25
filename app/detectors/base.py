from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
from PIL import Image


@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    class_names: list[str]
    def predict(self, image: Image.Image, confidence: float, iou: float, image_size: int) -> list[Detection]: ...
