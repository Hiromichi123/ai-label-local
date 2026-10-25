from __future__ import annotations

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class CropMode(StrEnum):
    center_crop = "center_crop"
    letterbox_pad = "letterbox_pad"


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w .-]+$")
    class_names: list[str] = Field(min_length=1)
    crop_mode: CropMode = CropMode.center_crop

    @field_validator("class_names")
    @classmethod
    def valid_classes(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values) or len(set(values)) != len(values):
            raise ValueError("class names must be non-empty and unique")
        return [value.strip() for value in values]


class Box(BaseModel):
    class_id: int = Field(ge=0)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(gt=0)
    y2: float = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    source: str = "human"
    status: str = "draft"

    @field_validator("x2")
    @classmethod
    def x2_positive(cls, value: float, info):
        if "x1" in info.data and value <= info.data["x1"]:
            raise ValueError("x2 must be greater than x1")
        return value

    @field_validator("y2")
    @classmethod
    def y2_positive(cls, value: float, info):
        if "y1" in info.data and value <= info.data["y1"]:
            raise ValueError("y2 must be greater than y1")
        return value


class AnnotationUpdate(BaseModel):
    version: int = Field(ge=0)
    boxes: list[Box]


class PredictRequest(BaseModel):
    model_path: str
    confidence: float = Field(default=0.25, ge=0, le=1)
    iou: float = Field(default=0.7, ge=0, le=1)
    image_size: int = Field(default=640, ge=32, le=4096)
    auto_accept: float | None = Field(default=None, ge=0, le=1)


class ExportRequest(BaseModel):
    train_ratio: float = Field(default=0.8, gt=0, lt=1)
    val_ratio: float = Field(default=0.1, ge=0, lt=1)
    seed: int = 42
    include_unreviewed: bool = False


class JobView(BaseModel):
    id: str
    kind: str
    status: str
    completed: int
    total: int
    errors: list[str] = []
    result: dict[str, Any] | None = None
