from __future__ import annotations

import json
import os
import random
import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, ImageOps
from .config import IMAGE_EXTENSIONS, MODEL_EXTENSIONS, within_workspace
from .detectors import load_detector
from .jobs import manager
from .schemas import AnnotationUpdate, ClassCreate, ClassRename, CropRequest, ExportRequest, PredictRequest, ProjectCreate
from .storage import Storage
from .schemas import ServerDirectoryImport
from .server_directories import roots, resolve_directory, image_files
from .labels import parse_yolo, load_label, class_ids, validate_label_text, model_class_mapping
from .schemas import LabelTextUpdate, TrainingRequest, ProjectRename
from . import model_management
from .training import TrainingManager, training_summary


router = APIRouter(prefix="/api")
storage = Storage()
training = TrainingManager()


def error(status: int, message: str):
    raise HTTPException(status_code=status, detail={"code": "request_error", "message": message})


@router.get("/config")
def config():
    return model_management.settings(storage)


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
        return storage.add_class(project_id, request.name, request.class_id)
    except FileNotFoundError:
        error(404, "project not found")
    except ValueError as exc:
        error(409, str(exc))


@router.put("/projects/{project_id}/classes/{class_id}")
def rename_class(project_id: str, class_id: int, request: ClassRename):
    try:
        return storage.rename_class(project_id, class_id, request.name, request.class_id)
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
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.post("/projects/{project_id}/images/{image_id}/crop")
def crop_image(project_id: str, image_id: str, request: CropRequest):
    try:
        return storage.crop_image(project_id, image_id, request.version, request.left, request.top)
    except (FileNotFoundError, KeyError):
        error(404, "image not found")
    except RuntimeError as exc:
        error(409, str(exc))
    except ValueError as exc:
        error(422, str(exc))


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


@router.get("/projects/{project_id}/images/{image_id}/label-file")
def label_file(project_id: str, image_id: str):
    try:
        record = storage.image_record(project_id, image_id)
        path = storage._project_path(project_id) / "labels" / f"{image_id}.txt"
        return {"version": record.get("version", 0), "filename": path.name,
                "text": path.read_text(encoding="utf-8") if path.exists() else ""}
    except FileNotFoundError:
        error(404, "image not found")


@router.post("/projects/{project_id}/images/{image_id}/label-file/validate")
def validate_label_file(project_id: str, image_id: str, request: LabelTextUpdate):
    try:
        record = storage.image_record(project_id,image_id)
        _, errors = validate_label_text(request.text,record['width'],record['height'],class_ids(storage.project(project_id)))
        return {'valid':not errors,'errors':errors}
    except FileNotFoundError:
        error(404,'image not found')


@router.put("/projects/{project_id}/images/{image_id}/label-file")
def update_label_file(project_id: str, image_id: str, request: LabelTextUpdate):
    try:
        record = storage.image_record(project_id, image_id)
        boxes, errors = validate_label_text(request.text, record['width'], record['height'], class_ids(storage.project(project_id)))
        if errors:
            first = errors[0]
            raise HTTPException(status_code=422,detail={'code':'invalid_labels','message':f"第 {first['line']} 行：{first['message']}",'errors':errors})
        return storage.save_annotations(project_id, image_id, request.version, boxes)
    except FileNotFoundError:
        error(404, "image not found")
    except RuntimeError as exc:
        error(409, str(exc))
    except ValueError as exc:
        error(422, str(exc))


@router.post("/projects/{project_id}/predict")
def predict(project_id: str, request: PredictRequest):
    project = storage.project(project_id)
    if request.scope == "current":
        if request.image_id not in project["images"]:
            error(404, "请先选择当前图片")
        if project["images"][request.image_id].get("needs_crop"):
            error(409, "请先确认当前图片的裁剪")
        image_ids = [request.image_id]
    else:
        image_ids = [iid for iid, record in project["images"].items()
                     if not record.get("needs_crop") and
                     (request.overwrite_existing or
                      (record.get("status") not in {"edited", "approved", "labeled"} and
                       (record.get("status") in {"predicted", "needs_review", "invalid"} or not storage.annotations(project_id, iid)["boxes"])))]
        if not image_ids:
            error(409, "没有可标注的未完成图片；已确认的标注保留，待裁剪图片需先裁剪")
    model_path = Path(request.model_path or model_management.settings(storage)["default_model_path"]).expanduser().resolve()
    if model_path.suffix.lower() not in MODEL_EXTENSIONS or not model_path.is_file():
        error(400, "model_path must point to an existing .pt or .onnx file")
    try:
        detector = load_detector(str(model_path))
        prediction_classes = model_class_mapping(getattr(detector,"class_names",[]),project)
    except Exception as exc:
        error(400, str(exc))

    def process(image_id: str):
        record = storage.image_record(project_id, image_id)
        image = Image.open(storage.image_path(project_id, image_id)).convert("RGB")
        detections = detector.predict(image, request.confidence, request.iou, request.image_size)
        from .labels import serialize
        from .schemas import Box
        boxes = [Box(class_id=prediction_classes[d.class_id], x1=max(0, d.x1), y1=max(0, d.y1), x2=min(record["width"], d.x2), y2=min(record["height"], d.y2), confidence=d.confidence, source="model", status="needs_review") for d in detections if d.class_id in prediction_classes and d.x2 > d.x1 and d.y2 > d.y1]
        # Never overwrite changes made while inference was running.
        with storage.lock:
            project_now = storage.project(project_id)
            if class_ids(project_now) != class_ids(project):
                raise RuntimeError("预测期间类别编号发生变化，未覆盖标注")
            record_now = project_now["images"][image_id]
            if record_now.get("version", 0) != record.get("version", 0):
                raise RuntimeError(f"{record['filename']} 在预测期间已修改，未覆盖")
            label = storage._project_path(project_id) / "labels" / f"{image_id}.txt"
            before = label.read_bytes()
            backup = storage._project_path(project_id) / "prediction_backups" / uuid.uuid4().hex
            backup.mkdir(parents=True)
            (backup / (image_id + ".txt")).write_bytes(before)
            (backup / "image.json").write_text(json.dumps(record_now, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                label.write_text(serialize(boxes, record["width"], record["height"]), encoding="utf-8")
                record_now.update({"box_count": len(boxes), "status": "predicted", "version": record_now.get("version", 0) + 1})
                storage._write_json(storage._project_path(project_id) / "project.json", project_now)
            except Exception:
                label.write_bytes(before)
                raise

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
            from .labels import serialize
            boxes = load_label(storage._project_path(project_id)/"labels"/f"{record['id']}.txt", record['width'], record['height'], class_ids(project))
            dense = {value:index for index,value in enumerate(class_ids(project))}
            boxes = [b.model_copy(update={'class_id':dense[b.class_id]}) for b in boxes]
            (output/"labels"/split/f"{record['id']}.txt").write_text(serialize(boxes,record['width'],record['height']),encoding='utf-8')
            manifest.append({"id": record["id"], "source": record["filename"], "split": split})
    (output / "dataset.yaml").write_text("path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames: " + json.dumps(project["class_names"], ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "class_mapping.json").write_text(json.dumps({"class_ids":class_ids(project),"class_names":project["class_names"]},ensure_ascii=False),encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(output), "counts": {split: len(items) for split, items in splits.items()}, "manifest": len(manifest)}


@router.get("/server/directories")
def server_directories(path: str = ""):
    allowed = roots(storage.root)
    if not path:
        return {"path": "", "parent": None, "directories": [{"name": str(p), "path": str(p)} for p in allowed], "image_count": 0}
    try:
        directory = resolve_directory(path, allowed)
        children = []
        count = 0
        for child in directory.iterdir():
            if child.name.startswith('.'):
                continue
            if child.is_dir():
                try:
                    resolved = resolve_directory(str(child), allowed)
                    children.append({"name": child.name, "path": str(resolved)})
                except (OSError, ValueError):
                    continue
            elif child.suffix.lower() in IMAGE_EXTENSIONS:
                count += 1
        parent = directory.parent
        return {"path": str(directory), "parent": str(parent) if parent != directory and any(parent.is_relative_to(p) for p in allowed) else "",
                "directories": sorted(children, key=lambda x: x["name"].lower()), "image_count": count}
    except PermissionError as exc:
        error(403, str(exc))
    except FileNotFoundError:
        error(404, "目录不存在")
    except (OSError, ValueError) as exc:
        error(422, str(exc))


@router.post("/server/directories/import")
def import_server_directory(request: ServerDirectoryImport):
    try:
        allowed = roots(storage.root)
        directory = resolve_directory(request.path, allowed)
        files = image_files(directory, allowed, request.recursive)
        if not files:
            raise ValueError("所选目录没有支持的图片")
        # Validate all labels first so a class mismatch never produces broken records.
        for source in files:
            try:
                with Image.open(source) as opened:
                    image = ImageOps.exif_transpose(opened)
                    width, height = image.size
                    image.load()
                label = source.with_suffix(".txt")
                if label.is_file():
                    load_label(label, width, height, request.class_ids if request.class_ids is not None else len(request.class_names))
            except (OSError, ValueError) as exc:
                raise ValueError(f"{source.name}: {exc}") from exc
        project = storage.create_project(request)
        try:
            imported = storage.import_files(project["id"], files)
        except Exception:
            shutil.rmtree(storage._project_path(project["id"]))
            raise
        return {"project_id": project["id"], "count": len(imported)}
    except PermissionError as exc:
        error(403, str(exc))
    except FileNotFoundError:
        error(404, "目录不存在")
    except (OSError, ValueError) as exc:
        error(422, str(exc))


@router.get("/training/environment")
def training_environment():
    import importlib.util
    devices = [{"id": "cpu", "name": "CPU"}]
    available = importlib.util.find_spec("ultralytics") is not None
    if available:
        try:
            import torch
            devices = [{"id": str(i), "name": f"GPU {i} · {torch.cuda.get_device_name(i)}"} for i in range(torch.cuda.device_count())] + devices
        except Exception:
            pass
    return {"available": available, "devices": devices}


@router.get("/projects/{project_id}/training")
def training_runs(project_id: str):
    try:
        return {"summary": training_summary(storage, project_id), "runs": training.list(storage, project_id)}
    except FileNotFoundError:
        error(404, "project not found")


@router.get("/projects/{project_id}/training-settings")
def training_settings(project_id: str):
    try:
        storage.project(project_id)
        path = storage._project_path(project_id)/"training_settings.json"
        if path.is_file():
            params = TrainingRequest.model_validate(json.loads(path.read_text(encoding='utf-8')))
            return {"params":params.model_dump(),"source":"saved"}
        runs = training.list(storage,project_id)
        if runs:
            return {"params":TrainingRequest.model_validate(runs[0]['params']).model_dump(),"source":"last_run"}
        params = TrainingRequest(model_path=model_management.settings(storage)['default_model_path'])
        return {"params":params.model_dump(),"source":"defaults"}
    except FileNotFoundError:
        error(404,"project not found")


@router.put("/projects/{project_id}/training-settings")
def save_training_settings(project_id: str, request: TrainingRequest):
    from .training import write_json
    try:
        with storage.lock:
            storage.project(project_id)
            write_json(storage._project_path(project_id)/"training_settings.json",request.model_dump())
        return {"params":request.model_dump(),"source":"saved"}
    except FileNotFoundError:
        error(404,"project not found")


@router.post("/projects/{project_id}/training")
def start_training(project_id: str, request: TrainingRequest):
    try:
        if not request.model_path:
            request = request.model_copy(update={"model_path":model_management.settings(storage)["default_model_path"]})
        save_training_settings(project_id,request)
        return training.start(storage, project_id, request)
    except FileNotFoundError as exc:
        error(404, str(exc))
    except RuntimeError as exc:
        error(409, str(exc))
    except (ValueError, ImportError) as exc:
        error(422, str(exc))


@router.get("/projects/{project_id}/training/{run_id}")
def training_run(project_id: str, run_id: str):
    try:
        return training.view(storage, project_id, run_id, logs=True)
    except (FileNotFoundError, ValueError):
        error(404, "training run not found")


@router.post("/projects/{project_id}/training/{run_id}/stop")
def stop_training(project_id: str, run_id: str):
    try:
        return training.stop(storage, project_id, run_id)
    except (FileNotFoundError, ValueError):
        error(404, "training run not found")


@router.put("/projects/{project_id}/name")
def rename_project(project_id: str, request: ProjectRename):
    try:
        return storage.rename_project(project_id, request.name)
    except FileNotFoundError:
        error(404,"project not found")
    except ValueError as exc:
        error(409,str(exc))


@router.get("/projects/{project_id}/images/{image_id}/download")
def download_image(project_id: str, image_id: str):
    try:
        record = storage.image_record(project_id,image_id)
        filename = Path(record['filename'].replace('\\','/')).stem + '.jpg'
        return FileResponse(storage.image_path(project_id,image_id),media_type='image/jpeg',filename=filename,headers={'Cache-Control':'no-store'})
    except FileNotFoundError:
        error(404,'image not found')


@router.get("/models")
def list_models():
    return model_management.list_models(storage,training)


@router.put("/models/{model_id}/default")
def default_model(model_id: str):
    try:
        return model_management.set_default(storage,training,model_id)
    except FileNotFoundError:
        error(404,'model not found')


@router.delete("/models/{model_id}")
def delete_model(model_id: str):
    try:
        return model_management.delete_model(storage,training,model_id)
    except FileNotFoundError:
        error(404,'model not found')
    except (ValueError,RuntimeError) as exc:
        error(409,str(exc))
