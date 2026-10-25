from __future__ import annotations

import json
import random
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image
from .config import IMAGE_EXTENSIONS, MODEL_EXTENSIONS, within_workspace
from .detectors import load_detector
from .jobs import manager
from .schemas import AnnotationUpdate, ClassCreate, ClassRename, ExportRequest, PredictRequest, ProjectCreate
from .storage import Storage


router = APIRouter(prefix="/api")
storage = Storage()


def error(status: int, message: str):
    raise HTTPException(status_code=status, detail={"code": "request_error", "message": message})


@router.get("/projects")
def projects():
    return storage.list_projects()


@router.post("/projects")
def create_project(request: ProjectCreate):
    try:
        return storage.create_project(request)
    except Exception as exc:
        error(400, str(exc))


@router.get("/projects/{project_id}")
def project(project_id: str):
    try:
        return storage.refresh_image_statuses(project_id)
    except FileNotFoundError:
        error(404, "project not found")


@router.post("/projects/{project_id}/classes")
def add_class(project_id: str, request: ClassCreate):
    try:
        return storage.add_class(project_id, request.name)
    except FileNotFoundError:
        error(404, "project not found")
    except ValueError as exc:
        error(409, str(exc))


@router.put("/projects/{project_id}/classes/{class_id}")
def rename_class(project_id: str, class_id: int, request: ClassRename):
    try:
        return storage.rename_class(project_id, class_id, request.name)
    except FileNotFoundError:
        error(404, "project not found")
    except IndexError as exc:
        error(404, str(exc))
    except ValueError as exc:
        error(409, str(exc))


@router.post("/projects/{project_id}/images/import")
async def import_images(project_id: str, files: list[UploadFile] = File(...)):
    storage.project(project_id)
    temporary = storage.root / "tmp" / uuid.uuid4().hex
    temporary.mkdir(parents=True, exist_ok=True)
    paths = []
    try:
        for upload in files:
            suffix = Path(upload.filename or "").suffix.lower()
            if suffix not in IMAGE_EXTENSIONS and suffix != ".txt":
                continue
            # Browser directory uploads include a relative path in filename.
            # Normalize it into a private temp tree; never use it as a project path.
            relative = Path(upload.filename or "").as_posix().lstrip("/")
            if not relative or relative.startswith("../") or "/../" in f"/{relative}":
                error(400, "invalid uploaded relative path")
            target = (temporary / relative).resolve()
            if not target.is_relative_to(temporary.resolve()):
                error(400, "invalid uploaded relative path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(await upload.read())
            paths.append(target)
        result = storage.import_files(project_id, paths, replace_existing=True)
        return {"images": result, "count": len(result), "replaced": True}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@router.get("/projects/{project_id}/images")
def images(project_id: str):
    return list(storage.project(project_id)["images"].values())


@router.get("/projects/{project_id}/images/{image_id}/asset")
def image_asset(project_id: str, image_id: str):
    try:
        path = storage.image_path(project_id, image_id)
    except FileNotFoundError:
        error(404, "image not found")
    return FileResponse(path, media_type="image/jpeg")


@router.get("/projects/{project_id}/images/{image_id}/annotation-status")
def annotation_status(project_id: str, image_id: str):
    try:
        return storage.annotation_status(project_id, image_id)
    except FileNotFoundError:
        error(404, "image not found")


@router.get("/projects/{project_id}/images/{image_id}/annotations")
def annotations(project_id: str, image_id: str):
    try:
        return storage.annotations(project_id, image_id)
    except FileNotFoundError:
        error(404, "image not found")
    except ValueError as exc:
        error(422, str(exc))


@router.put("/projects/{project_id}/images/{image_id}/annotations")
def update_annotations(project_id: str, image_id: str, request: AnnotationUpdate):
    try:
        return storage.save_annotations(project_id, image_id, request.version, request.boxes)
    except FileNotFoundError:
        error(404, "image not found")
    except RuntimeError as exc:
        error(409, str(exc))
    except ValueError as exc:
        error(422, str(exc))


@router.post("/projects/{project_id}/predict")
def predict(project_id: str, request: PredictRequest):
    project = storage.project(project_id)
    model_path = Path(request.model_path).expanduser().resolve()
    if model_path.suffix.lower() not in MODEL_EXTENSIONS or not model_path.is_file():
        error(400, "model_path must point to an existing .pt or .onnx file")
    image_ids = list(project["images"])
    try:
        detector = load_detector(str(model_path))
    except Exception as exc:
        error(400, str(exc))

    def process(image_id: str):
        record = storage.image_record(project_id, image_id)
        image = Image.open(storage.image_path(project_id, image_id)).convert("RGB")
        detections = detector.predict(image, request.confidence, request.iou, request.image_size)
        from .labels import serialize
        from .schemas import Box
        boxes = [Box(class_id=d.class_id, x1=max(0, d.x1), y1=max(0, d.y1), x2=min(record["width"], d.x2), y2=min(record["height"], d.y2), confidence=d.confidence, source="model", status="needs_review") for d in detections if d.x2 > d.x1 and d.y2 > d.y1]
        # Predictions are only drafts until a human saves or approves them.
        label = storage._project_path(project_id) / "labels" / f"{image_id}.txt"
        label.write_text(serialize(boxes, record["width"], record["height"]), encoding="utf-8")
        project_now = storage.project(project_id)
        record_now = project_now["images"][image_id]
        record_now.update({"box_count": len(boxes), "status": "predicted", "version": record_now.get("version", 0) + 1})
        storage._write_json(storage._project_path(project_id) / "project.json", project_now)

    return manager.submit("predict", image_ids, process).view()


@router.get("/jobs/{job_id}")
def job(job_id: str):
    try:
        return manager.get(job_id).view()
    except KeyError:
        error(404, "job not found")


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    try:
        return manager.cancel(job_id).view()
    except KeyError:
        error(404, "job not found")


@router.post("/projects/{project_id}/export")
def export_dataset(project_id: str, request: ExportRequest):
    project = storage.project(project_id)
    if request.train_ratio + request.val_ratio >= 1:
        error(422, "train_ratio + val_ratio must be less than 1")
    records = list(project["images"].values())
    eligible = [r for r in records if request.include_unreviewed or r.get("status") in {"approved", "edited"}]
    if not eligible:
        error(422, "no reviewed images available for export")
    rng = random.Random(request.seed)
    rng.shuffle(eligible)
    train_end = round(len(eligible) * request.train_ratio)
    val_end = train_end + round(len(eligible) * request.val_ratio)
    splits = {"train": eligible[:train_end], "val": eligible[train_end:val_end], "test": eligible[val_end:]}
    output = storage._project_path(project_id) / "exports" / f"dataset-{uuid.uuid4().hex[:8]}"
    for split in splits:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    for split, split_records in splits.items():
        for record in split_records:
            shutil.copy2(storage.image_path(project_id, record["id"]), output / "images" / split / f"{record['id']}.jpg")
            shutil.copy2(storage._project_path(project_id) / "labels" / f"{record['id']}.txt", output / "labels" / split / f"{record['id']}.txt")
            manifest.append({"id": record["id"], "source": record["filename"], "split": split})
    (output / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: " + json.dumps(project["class_names"], ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(output), "counts": {split: len(items) for split, items in splits.items()}, "manifest": len(manifest)}
