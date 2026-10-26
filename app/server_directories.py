from __future__ import annotations
import os
from pathlib import Path
from .config import IMAGE_EXTENSIONS


def roots(workspace: Path) -> list[Path]:
    configured = os.environ.get("YOLO_LABEL_IMPORT_ROOTS")
    paths = [Path(p).expanduser().resolve() for p in configured.split(os.pathsep) if p] if configured else [Path.home().resolve(), workspace.resolve()]
    return list(dict.fromkeys(p for p in paths if p.is_dir()))


def resolve_directory(value: str, allowed: list[Path]) -> Path:
    path = Path(value).expanduser().resolve(strict=True)
    if not any(path.is_relative_to(root) for root in allowed):
        raise PermissionError("目录不在允许浏览的范围内")
    if not path.is_dir():
        raise ValueError("请选择目录")
    return path


def image_files(path: Path, allowed: list[Path], recursive: bool) -> list[Path]:
    files = []
    for current, dirs, names in os.walk(path, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and not (Path(current)/d).is_symlink()) if recursive else []
        for name in sorted(names):
            source = Path(current)/name
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved = source.resolve()
            if not any(resolved.is_relative_to(root) for root in allowed):
                continue
            label = source.with_suffix('.txt')
            if label.exists() and not any(label.resolve().is_relative_to(root) for root in allowed):
                raise PermissionError("标签链接超出允许范围")
            files.append(source)
            if len(files) > 20000:
                raise ValueError("单次最多导入 20000 张图片，请选择较小目录")
    return files
