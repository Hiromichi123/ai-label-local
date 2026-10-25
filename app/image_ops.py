from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Transform:
    source_width: int
    source_height: int
    left: float
    top: float
    scale: float
    output_size: int

    def box(self, box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
        x1, y1, x2, y2 = box
        x1 = (x1 - self.left) * self.scale
        y1 = (y1 - self.top) * self.scale
        x2 = (x2 - self.left) * self.scale
        y2 = (y2 - self.top) * self.scale
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(self.output_size, x2), min(self.output_size, y2)
        return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def square_transform(width: int, height: int, mode: str, output_size: int | None = None) -> Transform:
    size = output_size or max(width, height)
    if mode == "center_crop":
        side = min(width, height)
        return Transform(width, height, (width - side) / 2, (height - side) / 2, size / side, size)
    if mode == "letterbox_pad":
        scale = size / max(width, height)
        return Transform(width, height, 0, 0, scale, size)
    raise ValueError(f"unsupported crop mode: {mode}")


def preprocess(source: Path, destination: Path, mode: str) -> Transform:
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        transform = square_transform(*image.size, mode)
        if mode == "center_crop":
            side = min(image.size)
            left = int(round((image.width - side) / 2))
            top = int(round((image.height - side) / 2))
            image = image.crop((left, top, left + side, top + side))
        else:
            scale = transform.scale
            resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
            image = Image.new("RGB", (transform.output_size, transform.output_size), (114, 114, 114))
            image.paste(resized, (0, 0))
        destination.parent.mkdir(parents=True, exist_ok=True)
        image.resize((transform.output_size, transform.output_size), Image.Resampling.LANCZOS).save(destination, quality=95)
        return transform
