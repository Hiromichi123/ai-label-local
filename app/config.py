from __future__ import annotations

import os
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MODEL_EXTENSIONS = {".pt", ".onnx"}


def workspace_root() -> Path:
    root = Path(os.environ.get("YOLO_LABEL_WORKSPACE", "./workspace")).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def within_workspace(path: Path, root: Path | None = None) -> Path:
    root = (root or workspace_root()).resolve()
    resolved = path.expanduser().resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path must remain inside the configured workspace")
    return resolved
