from __future__ import annotations

import json
import re
import shutil
import uuid
import threading
from functools import wraps
from datetime import datetime
from PIL import Image, ImageOps
from pathlib import Path
from .config import IMAGE_EXTENSIONS, within_workspace, workspace_root
from .image_ops import preprocess
from .labels import load_label, serialize, class_ids
from .schemas import Box, CropMode, ProjectCreate


def locked(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped


class Storage:
    def __init__(self, root: Path | None = None):
        self.lock = threading.RLock()
        self.root = (root or workspace_root()).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _project_path(self, project_id: str) -> Path:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{8,64}", project_id):
            raise ValueError("invalid project id")
        return within_workspace(self.root / "projects" / project_id, self.root)

    def project(self, project_id: str) -> dict:
        path = self._project_path(project_id) / "project.json"
        if not path.is_file():
            raise FileNotFoundError("project not found")
        return json.loads(path.read_text(encoding="utf-8"))

    @locked
    def create_project(self, request: ProjectCreate) -> dict:
        project_id = uuid.uuid4().hex
        path = self._project_path(project_id)
        (path / "images").mkdir(parents=True)
        (path / "labels").mkdir()
        (path / "originals").mkdir()
        metadata = {"id": project_id, "name": request.name, "class_names": request.class_names,
                    "class_ids": request.class_ids if request.class_ids is not None else list(range(len(request.class_names))),
                    "crop_mode": request.crop_mode.value, "version": 0, "images": {}}
        self._write_json(path / "project.json", metadata)
        return metadata

    @staticmethod
    def _write_json(path: Path, value: object) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)

    def list_projects(self) -> list[dict]:
        result = []
        for path in (self.root / "projects").glob("*/project.json"):
            try:
                result.append(json.loads(path.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue
        return sorted(result, key=lambda item: item["name"].lower())

    @locked
    def refresh_image_statuses(self, project_id: str) -> dict:
        project = self.project(project_id)
        class_count = class_ids(project)
        changed = False
        for record in project["images"].values():
            label_path = self._project_path(project_id) / "labels" / f"{record['id']}.txt"
            try:
                boxes = load_label(label_path, record["width"], record["height"], class_count)
                count = len(boxes)
                previous = record.get("status")
                status = previous if previous in {"edited", "approved", "predicted"} else ("labeled" if count > 0 else "unlabeled")
                if record.get("needs_crop"):
                    status = "needs_crop"
            except (OSError, ValueError):
                status = "invalid"
                count = record.get("box_count", 0)
            if record.get("status") != status or record.get("box_count") != count:
                record["status"] = status
                record["box_count"] = count
                changed = True
        if changed:
            project["version"] += 1
            self._write_json(self._project_path(project_id) / "project.json", project)
        return project

    @locked
    def rename_project(self, project_id: str, name: str) -> dict:
        project = self.project(project_id)
        if any(p['id'] != project_id and p['name'] == name for p in self.list_projects()):
            raise ValueError('项目名称已存在')
        project['name'] = name.strip()
        if not project['name']:
            raise ValueError('项目名称不能为空')
        project['version'] += 1
        self._write_json(self._project_path(project_id)/'project.json',project)
        return project

    @locked
    def add_class(self, project_id: str, name: str, class_id: int | None = None) -> dict:
        project = self.project(project_id)
        name = name.strip()
        ids = class_ids(project)
        value = class_id if class_id is not None else max(ids, default=-1)+1
        if not name or name in project['class_names']:
            raise ValueError('类别名为空或已存在')
        if value in ids or not 0 <= value <= 999999:
            raise ValueError('类别编号已存在或超出范围')
        project['class_names'].append(name)
        project['class_ids'] = ids+[value]
        project['version'] += 1
        self._write_json(self._project_path(project_id)/'project.json',project)
        return {'class_id':value,'name':name,'class_names':project['class_names'],'class_ids':project['class_ids'],'version':project['version']}

    @locked
    def rename_class(self, project_id: str, class_id: int, name: str, new_id: int | None = None) -> dict:
        project = self.project(project_id)
        ids = class_ids(project)
        if class_id not in ids:
            raise IndexError('class id outside project classes')
        index = ids.index(class_id)
        value = class_id if new_id is None else new_id
        name = name.strip()
        if not name or any(n == name and i != index for i,n in enumerate(project['class_names'])):
            raise ValueError('类别名为空或已存在')
        if value != class_id and value in ids or not 0 <= value <= 999999:
            raise ValueError('类别编号已存在或超出范围')
        base = self._project_path(project_id)
        changes = {}
        if value != class_id:
            for iid,record in project['images'].items():
                path = base/'labels'/f'{iid}.txt'
                text = path.read_text(encoding='utf-8')
                # Validate before changing anything; preserve original coordinate precision.
                load_label(path,record['width'],record['height'],ids)
                lines = text.splitlines(keepends=True)
                changed = False
                for i,line in enumerate(lines):
                    fields = line.split()
                    if fields and int(fields[0]) == class_id:
                        lines[i] = re.sub(r'^(\s*)\d+',lambda m:m.group(1)+str(value),line)
                        changed = True
                if changed:
                    changes[path] = ''.join(lines)
                    record['version'] = record.get('version',0)+1
        project['class_names'][index] = name
        ids[index] = value
        project['class_ids'] = ids
        project['version'] += 1
        backup = base/'class_backups'/uuid.uuid4().hex
        backup.mkdir(parents=True)
        metadata = base/'project.json'
        shutil.copy2(metadata,backup/'project.json')
        for path in changes:
            shutil.copy2(path,backup/path.name)
        try:
            for path,text in changes.items():
                path.write_text(text,encoding='utf-8')
            self._write_json(metadata,project)
        except Exception:
            for path in changes:
                shutil.copy2(backup/path.name,path)
            shutil.copy2(backup/'project.json',metadata)
            raise
        return {'class_id':value,'name':name,'class_names':project['class_names'],'class_ids':ids,'version':project['version']}

    @locked
    def clear_images(self, project_id: str) -> None:
        project = self.project(project_id)
        base = self._project_path(project_id)
        for directory in (base / "images", base / "labels", base / "originals"):
            directory.mkdir(parents=True, exist_ok=True)
            for child in directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        project["images"] = {}
        project["version"] += 1
        self._write_json(base / "project.json", project)

    @locked
    def import_files(self, project_id: str, files: list[Path], replace_existing: bool = False) -> list[dict]:
        project = self.project(project_id)
        base = self._project_path(project_id)
        image_files = [Path(source) for source in files if Path(source).suffix.lower() in IMAGE_EXTENSIONS]
        if replace_existing:
            self.clear_images(project_id)
            project = self.project(project_id)
            base = self._project_path(project_id)
        imported = []
        for source in image_files:
            source = Path(source)
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = uuid.uuid4().hex
            relative_source = source.relative_to(self.root / "tmp") if source.is_relative_to(self.root / "tmp") else Path(source.name)
            original = base / "originals" / f"{image_id}{source.suffix.lower()}"
            image_path = base / "images" / f"{image_id}.jpg"
            label_path = base / "labels" / f"{image_id}.txt"
            shutil.copy2(source, original)
            # Import the full image. Cropping is an explicit, persistent next step.
            with Image.open(original) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
                width, height = image.size
                image.save(image_path, format="JPEG", quality=95)
            boxes = []
            sibling = source.with_suffix(".txt")
            if sibling.is_file():
                try:
                    boxes = load_label(sibling, width, height, class_ids(project))
                except ValueError as exc:
                    project["images"][image_id] = {"id": image_id, "status": "invalid", "error": str(exc)}
                    continue
            label_path.write_text(serialize(boxes, width, height), encoding="utf-8")
            record = {"id": image_id, "filename": source.name, "source_path": relative_source.as_posix(), "path": f"images/{image_id}.jpg", "width": width, "height": height,
                      "needs_crop": width != height, "status": "needs_crop" if width != height else ("approved" if boxes else "unlabeled"), "version": 0, "box_count": len(boxes)}
            project["images"][image_id] = record
            imported.append(record)
        project["version"] += 1
        self._write_json(base / "project.json", project)
        return imported

    def image_record(self, project_id: str, image_id: str) -> dict:
        project = self.project(project_id)
        if image_id not in project["images"]:
            raise FileNotFoundError("image not found")
        return project["images"][image_id]

    def image_path(self, project_id: str, image_id: str) -> Path:
        self.image_record(project_id, image_id)
        return self._project_path(project_id) / "images" / f"{image_id}.jpg"

    def annotations(self, project_id: str, image_id: str) -> dict:
        record = self.image_record(project_id, image_id)
        label_path = self._project_path(project_id) / "labels" / f"{image_id}.txt"
        boxes = load_label(label_path, record["width"], record["height"], class_ids(self.project(project_id)))
        return {"version": record.get("version", 0), "boxes": [box.model_dump() for box in boxes]}

    def annotation_status(self, project_id: str, image_id: str) -> dict:
        project = self.project(project_id)
        record = self.image_record(project_id, image_id)
        label_path = self._project_path(project_id) / "labels" / f"{image_id}.txt"
        if not label_path.is_file():
            return {"saved": False, "valid": False, "file_exists": False, "box_count": 0,
                    "version": record.get("version", 0), "status": "missing"}
        try:
            boxes = load_label(label_path, record["width"], record["height"], class_ids(project))
        except (OSError, ValueError) as exc:
            return {"saved": True, "valid": False, "file_exists": True, "box_count": None,
                    "version": record.get("version", 0), "status": "invalid", "error": str(exc)}
        metadata_count = record.get("box_count")
        metadata_consistent = metadata_count == len(boxes)
        # The label file is the source of truth for whether annotations were
        # saved correctly. Metadata can be stale after an interrupted write.
        has_labels = len(boxes) > 0
        return {"saved": True, "valid": True, "file_exists": True, "box_count": len(boxes),
                "metadata_consistent": metadata_consistent,
                "version": record.get("version", 0), "status": record.get("status", "saved") if has_labels else "unlabeled",
                "error": None if metadata_consistent else "project metadata is stale; label file is valid"}

    @locked
    def save_annotations(self, project_id: str, image_id: str, version: int, boxes: list[Box]) -> dict:
        project = self.project(project_id)
        if image_id not in project["images"]:
            raise FileNotFoundError("image not found")
        record = project["images"][image_id]
        if record.get("needs_crop"):
            raise RuntimeError("请先确认永久裁剪，再保存标注")
        if record.get("version", 0) != version:
            raise RuntimeError("annotation version conflict")
        for box in boxes:
            if box.class_id not in class_ids(project):
                raise ValueError("class id outside project classes")
            if box.x2 > record["width"] or box.y2 > record["height"]:
                raise ValueError("box outside image")
        path = self._project_path(project_id) / "labels" / f"{image_id}.txt"
        path.write_text(serialize(boxes, record["width"], record["height"]), encoding="utf-8")
        record["version"] = version + 1
        record["box_count"] = len(boxes)
        record["status"] = "edited"
        project["version"] += 1
        self._write_json(self._project_path(project_id) / "project.json", project)
        return {"version": record["version"], "boxes": [box.model_dump() for box in boxes]}

    @locked
    def crop_image(self, project_id: str, image_id: str, version: int, left: int, top: int) -> dict:
        project = self.project(project_id)
        record = project["images"][image_id]
        if record.get("version", 0) != version:
            raise RuntimeError("图片已被修改，请重新打开后裁剪")
        if not record.get("needs_crop"):
            raise ValueError("这张图已为正方形，无需重复裁剪")
        width, height = record["width"], record["height"]
        side = min(width, height)
        if left < 0 or top < 0 or left + side > width or top + side > height:
            raise ValueError("裁剪区域超出图像边界")
        base = self._project_path(project_id)
        image_path = self.image_path(project_id, image_id)
        label_path = base / "labels" / f"{image_id}.txt"
        boxes = load_label(label_path, width, height, class_ids(project))
        cropped_boxes = []
        for box in boxes:
            x1, y1 = max(0, box.x1 - left), max(0, box.y1 - top)
            x2, y2 = min(side, box.x2 - left), min(side, box.y2 - top)
            if x2 > x1 and y2 > y1:
                cropped_boxes.append(box.model_copy(update={"x1": x1, "y1": y1, "x2": x2, "y2": y2}))
        with Image.open(image_path) as image:
            if image.size != (width, height):
                raise ValueError("图像尺寸与索引不一致")
            cropped = image.crop((left, top, left + side, top + side))
        backup = base / "crop_backups" / (image_id + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        backup.mkdir(parents=True)
        for p in (image_path, label_path, base / "project.json"):
            shutil.copy2(p, backup / p.name)
        temporary = image_path.with_suffix(".crop.tmp")
        try:
            cropped.save(temporary, format="JPEG", quality=95)
            temporary.replace(image_path)
            label_path.write_text(serialize(cropped_boxes, side, side), encoding="utf-8")
            record.update(width=side, height=side, needs_crop=False, version=version + 1,
                          box_count=len(cropped_boxes), status="unlabeled" if not cropped_boxes else "approved",
                          crop={"left": left, "top": top, "side": side, "source_width": width, "source_height": height})
            project["version"] += 1
            self._write_json(base / "project.json", project)
        except Exception:
            for p in (image_path, label_path, base / "project.json"):
                shutil.copy2(backup / p.name, p)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return {"image": record, "version": record["version"], "boxes": [b.model_dump() for b in cropped_boxes]}
