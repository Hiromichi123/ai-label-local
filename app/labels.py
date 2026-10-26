from __future__ import annotations

from pathlib import Path
import math
from .schemas import Box


def parse_yolo(text: str, width: int, height: int, class_count: int | list[int]) -> list[Box]:
    allowed = range(class_count) if isinstance(class_count, int) else class_count
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
        if class_id not in allowed or not 0 <= cx <= 1 or not 0 <= cy <= 1 or not 0 < bw <= 1 or not 0 < bh <= 1:
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


def load_label(path: Path, width: int, height: int, class_count: int | list[int]) -> list[Box]:
    return parse_yolo(path.read_text(encoding="utf-8") if path.exists() else "", width, height, class_count)


def class_ids(project: dict) -> list[int]:
    return project.get("class_ids", list(range(len(project["class_names"]))))


def validate_label_text(text: str, width: int, height: int, allowed: list[int]):
    errors = []
    boxes = []
    for number, line in enumerate(text.split('\n'), 1):
        if not line.strip():
            continue
        fields = line.split()
        message = None
        if len(fields) != 5:
            message = '每行应有 5 列：类别编号 中心X 中心Y 宽 高'
        else:
            try:
                value = int(fields[0])
            except ValueError:
                message = '类别编号必须为整数'
            else:
                if value not in allowed:
                    message = f'类别编号 {value} 未在当前项目定义'
            if message is None:
                try:
                    cx, cy, bw, bh = map(float, fields[1:])
                except ValueError:
                    message = '坐标必须为数字'
                else:
                    if not all(math.isfinite(v) for v in (cx,cy,bw,bh)):
                        message = '坐标必须为有限数字，不能是 NaN 或 Infinity'
                    elif not 0 <= cx <= 1 or not 0 <= cy <= 1:
                        message = '中心坐标必须在 0–1 之间'
                    elif not 0 < bw <= 1 or not 0 < bh <= 1:
                        message = '宽和高必须大于 0 且不超过 1'
                    elif cx-bw/2 < -1e-6 or cy-bh/2 < -1e-6 or cx+bw/2 > 1+1e-6 or cy+bh/2 > 1+1e-6:
                        message = '标注框超出图片边界'
        if message is None:
            try:
                boxes.extend(parse_yolo(line,width,height,allowed))
            except ValueError:
                message = '标注框坐标无效'
        if message:
            errors.append({'line':number,'message':message})
    return boxes, errors


def model_class_mapping(model_names: list[str], project: dict, fallback_class_id: int | None = None) -> dict[int, int]:
    ids = class_ids(project)
    if fallback_class_id is not None and fallback_class_id not in ids:
        raise ValueError('目标类别编号不在当前项目中')
    if not model_names:
        return dict(enumerate(ids))
    missing = [name for name in model_names if name not in project['class_names']]
    if missing and fallback_class_id is None:
        raise ValueError('模型类别不在当前项目中：' + '、'.join(missing) + '。请切换对应项目或先新增这些类别。')
    return {i:ids[project['class_names'].index(name)] if name in project['class_names'] else fallback_class_id for i,name in enumerate(model_names)}
