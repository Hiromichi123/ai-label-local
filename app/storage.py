from __future__ import annotations

import json
import re
import shutil
import uuid
from pathlib import Path
from .config import IMAGE_EXTENSIONS, within_workspace, workspace_root
from .image_ops import preprocess
from .labels import load_label, serialize
from .schemas import Box, CropMode, ProjectCreate


class Storage:
    def __init__(self, root: Path | None = None):
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

    def create_project(self, request: ProjectCreate) -> dict:
        project_id = uuid.uuid4().hex
        path = self._project_path(project_id)
        (path / "images").mkdir(parents=True)
        (path / "labels").mkdir()
        (path / "originals").mkdir()
        metadata = {"id": project_id, "name": request.name, "class_names": request.class_names,
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

    def import_files(self, project_id: str, files: list[Path]) -> list[dict]:
        project = self.project(project_id)
        base = self._project_path(project_id)
        imported = []
        for source in files:
            source = Path(source)
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image_id = uuid.uuid4().hex
            original = base / "originals" / f"{image_id}{source.suffix.lower()}"
            image_path = base / "images" / f"{image_id}.jpg"
            label_path = base / "labels" / f"{image_id}.txt"
            shutil.copy2(source, original)
            transform = preprocess(original, image_path, project["crop_mode"])
            boxes = []
            side = transform.output_size
            sibling = source.with_suffix(".txt")
            if sibling.is_file():
                try:
                    source_width, source_height = transform.source_width, transform.source_height
                    for box in load_label(sibling, source_width, source_height, len(project["class_names"])):
                        mapped = transform.box((box.x1, box.y1, box.x2, box.y2))
                        if mapped:
                            boxes.append(Box(class_id=box.class_id, x1=mapped[0], y1=mapped[1], x2=mapped[2], y2=mapped[3], source="file", status="draft"))
                except ValueError as exc:
                    project["images"][image_id] = {"id": image_id, "status": "invalid", "error": str(exc)}
                    continue
            label_path.write_text(serialize(boxes, side, side), encoding="utf-8")
            record = {"id": image_id, "filename": source.name, "path": f"images/{image_id}.jpg", "width": side, "height": side,
                      "status": "approved" if boxes else "unlabeled", "version": 0, "box_count": len(boxes)}
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
        boxes = load_label(label_path, record["width"], record["height"], len(self.project(project_id)["class_names"]))
        return {"version": record.get("version", 0), "boxes": [box.model_dump() for box in boxes]}

    def annotation_status(self, project_id: str, image_id: str) -> dict:
        project = self.project(project_id)
        record = self.image_record(project_id, image_id)
        label_path = self._project_path(project_id) / "labels" / f"{image_id}.txt"
        if not label_path.is_file():
            return {"saved": False, "valid": False, "file_exists": False, "box_count": 0,
                    "version": record.get("version", 0), "status": "missing"}
        try:
            boxes = load_label(label_path, record["width"], record["height"], len(project["class_names"]))
        except (OSError, ValueError) as exc:
            return {"saved": True, "valid": False, "file_exists": True, "box_count": None,
                    "version": record.get("version", 0), "status": "invalid", "error": str(exc)}
        metadata_count = record.get("box_count")
        valid = metadata_count == len(boxes)
        return {"saved": True, "valid": valid, "file_exists": True, "box_count": len(boxes),
                "version": record.get("version", 0), "status": record.get("status", "saved") if valid else "mismatch",
                "error": None if valid else "project metadata does not match label file"}

    def save_annotations(self, project_id: str, image_id: str, version: int, boxes: list[Box]) -> dict:
        project = self.project(project_id)
        record = self.image_record(project_id, image_id)
        if record.get("version", 0) != version:
            raise RuntimeError("annotation version conflict")
        for box in boxes:
            if box.class_id >= len(project["class_names"]):
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
