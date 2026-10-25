from __future__ import annotations

from pathlib import Path
from .schemas import Box


def parse_yolo(text: str, width: int, height: int, class_count: int) -> list[Box]:
    boxes: list[Box] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"line {line_number}: expected class cx cy width height")
        try:
            class_id = int(fields[0])
            cx, cy, bw, bh = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"line {line_number}: non-numeric label") from exc
        if not 0 <= class_id < class_count or not 0 <= cx <= 1 or not 0 <= cy <= 1 or not 0 < bw <= 1 or not 0 < bh <= 1:
            raise ValueError(f"line {line_number}: values outside YOLO bounds")
        x1, x2 = (cx - bw / 2) * width, (cx + bw / 2) * width
        y1, y2 = (cy - bh / 2) * height, (cy + bh / 2) * height
        boxes.append(Box(class_id=class_id, x1=max(0, x1), y1=max(0, y1), x2=min(width, x2), y2=min(height, y2), source="file", status="draft"))
    return boxes


def to_yolo(box: Box, width: int, height: int) -> str:
    cx = ((box.x1 + box.x2) / 2) / width
    cy = ((box.y1 + box.y2) / 2) / height
    bw, bh = (box.x2 - box.x1) / width, (box.y2 - box.y1) / height
    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
        raise ValueError("box is outside image bounds")
    return f"{box.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def serialize(boxes: list[Box], width: int, height: int) -> str:
    return "\n".join(to_yolo(box, width, height) for box in boxes) + ("\n" if boxes else "")


def load_label(path: Path, width: int, height: int, class_count: int) -> list[Box]:
    return parse_yolo(path.read_text(encoding="utf-8") if path.exists() else "", width, height, class_count)
