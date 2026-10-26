from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class CropMode(StrEnum):
    manual_crop = "manual_crop"
    center_crop = "center_crop"
    letterbox_pad = "letterbox_pad"


class ProjectCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w .-]+$")
    class_names: list[str] = Field(min_length=1)
    class_ids: list[int] | None = None
    crop_mode: CropMode = CropMode.manual_crop

    @model_validator(mode="after")
    def validate_ids(self):
        if self.class_ids is not None and (len(self.class_ids) != len(self.class_names) or len(set(self.class_ids)) != len(self.class_ids) or any(i < 0 or i > 999999 for i in self.class_ids)):
            raise ValueError("类别编号必须唯一、非负，且与类别名称一一对应")
        return self

    @field_validator("class_names")
    @classmethod
    def valid_classes(cls, values: list[str]) -> list[str]:
        values = [value.strip() for value in values]
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError("class names must be non-empty and unique")
        return [value.strip() for value in values]


class ClassCreate(BaseModel):
    class_id: int | None = Field(default=None, ge=0, le=999999)
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("class name must not be blank")
        return value


class ClassRename(BaseModel):
    class_id: int | None = Field(default=None, ge=0, le=999999)
    name: str = Field(min_length=1, max_length=80)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("class name must not be blank")
        return value



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


class CropRequest(BaseModel):
    version: int = Field(ge=0)
    left: int = Field(ge=0)
    top: int = Field(ge=0)


class PredictRequest(BaseModel):
    target_class_id: int | None = Field(default=None, ge=0, le=999999)
    model_path: str = ""
    overwrite_existing: bool = False
    scope: Literal["unfinished", "current"] = "unfinished"
    image_id: str | None = None
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


class LabelTextUpdate(BaseModel):
    version: int = Field(ge=0)
    text: str


class ServerDirectoryImport(ProjectCreate):
    path: str
    recursive: bool = True



class TrainingRequest(BaseModel):
    name: str = Field(default="标注模型", min_length=1, max_length=80)
    model_path: str = ""
    epochs: int = Field(default=100, ge=1, le=2000)
    imgsz: int = Field(default=960, ge=64, le=2048)
    batch: int = Field(default=4, ge=1, le=128)
    device: str = Field(default="0", pattern=r"^(cpu|[0-9]+)$")
    lr0: float = Field(default=0.001, gt=0, le=1)
    patience: int = Field(default=30, ge=0, le=2000)
    val_ratio: float = Field(default=0.2, gt=0, lt=1)
    seed: int = Field(default=42, ge=0, le=2147483647)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    fliplr: float = Field(default=0.5, ge=0, le=1)
    mosaic: float = Field(default=1, ge=0, le=1)

    @field_validator("imgsz")
    @classmethod
    def stride_size(cls, value):
        if value % 32:
            raise ValueError("训练分辨率必须是 32 的倍数")
        return value


class ProjectRename(BaseModel):
    name: str = Field(min_length=1, max_length=80, pattern=r"^[\w .-]+$")
