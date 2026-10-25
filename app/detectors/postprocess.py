from __future__ import annotations
import numpy as np
from .base import Detection


def nms(outputs, width: int, height: int, confidence: float, iou: float) -> list[Detection]:
    raw = np.asarray(outputs[0])
    if raw.ndim == 3:
        raw = raw[0]
    if raw.ndim != 2:
        raise ValueError("unsupported ONNX output rank; configure an adapter for this model")
    if raw.shape[0] < raw.shape[1] and raw.shape[0] <= 256:
        raw = raw.T
    if raw.shape[1] < 6:
        raise ValueError("unsupported ONNX detection output; expected xywh + class scores")
    result = []
    for row in raw:
        cx, cy, w, h = map(float, row[:4])
        scores = row[4:]
        class_id = int(np.argmax(scores))
        score = float(scores[class_id])
        if score < confidence:
            continue
        # Most exports use input-space pixels; normalized outputs are handled too.
        if max(cx, cy, w, h) <= 2:
            cx, cy, w, h = cx * width, cy * height, w * width, h * height
        result.append(Detection(class_id, score, max(0, cx - w / 2), max(0, cy - h / 2), min(width, cx + w / 2), min(height, cy + h / 2)))
    return result
