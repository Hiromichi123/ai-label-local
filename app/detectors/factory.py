from __future__ import annotations
from pathlib import Path
from .base import Detector


def load_detector(path: str, device: str = "cpu") -> Detector:
    model_path = Path(path).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if model_path.suffix.lower() == ".pt":
        from .ultralytics_detector import UltralyticsDetector
        return UltralyticsDetector(model_path, device)
    if model_path.suffix.lower() == ".onnx":
        from .onnx_detector import OnnxDetector
        return OnnxDetector(model_path)
    raise ValueError("only .pt and .onnx models are supported")
