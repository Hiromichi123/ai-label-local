from __future__ import annotations
from pathlib import Path
from PIL import Image
from .base import Detection


class UltralyticsDetector:
    def __init__(self, path: Path, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("install the 'pt' extra to use .pt models") from exc
        self.model = YOLO(str(path))
        self.device = device
        names = self.model.names
        self.class_names = [names[index] for index in sorted(names)] if isinstance(names, dict) else list(names)

    def predict(self, image: Image.Image, confidence: float, iou: float, image_size: int) -> list[Detection]:
        result = self.model.predict(source=image, conf=confidence, iou=iou, imgsz=image_size, device=self.device, verbose=False)[0]
        if result.boxes is None:
            return []
        detections = []
        for box, score, class_id in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
            detections.append(Detection(int(class_id), float(score), *map(float, box)))
        return detections
