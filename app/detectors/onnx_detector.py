from __future__ import annotations
from pathlib import Path
from PIL import Image
from .base import Detection


class OnnxDetector:
    """Adapter for common YOLO ONNX exports; custom layouts fail explicitly."""
    def __init__(self, path: Path):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("install the 'onnx' extra to use .onnx models") from exc
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        self.image_size = int(shape[-1]) if isinstance(shape[-1], int) else 640
        self.class_names = []

    def predict(self, image: Image.Image, confidence: float, iou: float, image_size: int) -> list[Detection]:
        import cv2
        import numpy as np
        from .postprocess import nms
        rgb = np.asarray(image.convert("RGB"))
        resized = cv2.resize(rgb, (image_size, image_size)).astype("float32") / 255
        output = self.session.run(None, {self.input_name: resized.transpose(2, 0, 1)[None]})
        return nms(output, image.width, image.height, confidence, iou)
