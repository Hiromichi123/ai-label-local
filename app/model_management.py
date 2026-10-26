from __future__ import annotations
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone
from .training import read_json, write_json, ACTIVE


def settings(storage):
    values = read_json(storage.root/'model_settings.json', {})
    return {'default_model_path': values.get('default_model_path',os.environ.get('YOLO_LABEL_DEFAULT_MODEL','')),
            'default_image_size': values.get('default_image_size',int(os.environ.get('YOLO_LABEL_DEFAULT_IMAGE_SIZE','640')))}


def identity(path):
    return hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:32]


def list_models(storage, training):
    default = settings(storage)['default_model_path']
    default_path = Path(default).expanduser().resolve() if default else None
    active_paths = set()
    for metadata in (storage.root/'training').glob('*/run.json'):
        run = read_json(metadata,{})
        if run.get('status') in ACTIVE:
            active_paths.add(str(Path(run['params']['model_path']).resolve()))
    models = []
    projects = {p["id"]:p["name"] for p in storage.list_projects()}
    for metadata in (storage.root/'training').glob('*/run.json'):
        run = read_json(metadata,{})
        if run.get('status') != 'completed' or run.get('model_deleted'):
            continue
        path = Path(run.get('model_path','')).resolve()
        run_root = metadata.parent.resolve()
        if not path.is_file() or not path.is_relative_to(run_root) or path.name != 'best.pt':
            continue
        files = [p for p in [path,path.with_name('last.pt')] if p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(run_root)]
        if path not in files:
            continue
        blocked = '默认模型，请先将其他模型设为默认' if any(p.resolve()==default_path for p in files) else ('正在训练使用，暂不能删除' if any(str(p.resolve()) in active_paths for p in files) else '')
        models.append({'id':identity(path),'name':run['name'],'path':str(path),'kind':'training','project_id':run['project_id'],
                       'class_names':run['class_names'],'class_ids':run.get('class_ids',list(range(len(run['class_names'])))),
                       'imgsz':run['params']['imgsz'],'bytes':sum(p.stat().st_size for p in files),'default':path==default_path,
                       'delete_blocked':blocked,'run_id':run['id'],'project_name':projects.get(run['project_id'],run['project_id']),'epochs':run['params'].get('epochs'),'created_at':run.get('created_at','')})
    model_root = (storage.root.parent/'models').resolve()
    for candidate in model_root.rglob('*') if model_root.is_dir() else []:
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix.lower() not in {'.pt','.onnx'}:
            continue
        path = candidate.resolve()
        if not path.is_relative_to(model_root):
            continue
        blocked = '默认模型，请先将其他模型设为默认' if path==default_path else ('正在训练使用，暂不能删除' if str(path) in active_paths else '')
        models.append({'id':identity(path),'name':str(path.relative_to(model_root)),'path':str(path),'kind':'file','bytes':path.stat().st_size,
                       'default':path==default_path,'delete_blocked':blocked,'imgsz':settings(storage)['default_image_size']})
    return {'models':models,'default_model_path':default}


def find_model(storage, training, model_id):
    entry = next((m for m in list_models(storage,training)['models'] if m['id']==model_id),None)
    if entry is None:
        raise FileNotFoundError('model not found')
    return entry


def set_default(storage, training, model_id):
    with training.lock:
        model = find_model(storage,training,model_id)
        write_json(storage.root/'model_settings.json',{'default_model_path':model['path'],'default_image_size':model['imgsz']})
        return settings(storage)


def delete_model(storage, training, model_id):
    with training.lock:
        model = find_model(storage,training,model_id)
        if model['delete_blocked']:
            raise RuntimeError(model['delete_blocked'])
        path = Path(model['path'])
        files = [path]
        if model['kind']=='training':
            files += [path.with_name('last.pt')]
        # Move weights into a private staging directory before deleting, allowing rollback
        # if metadata cannot be saved. Dataset snapshots, labels and logs stay untouched.
        trash = storage.root/'model-trash'/os.urandom(16).hex()
        trash.mkdir(parents=True)
        moved = []
        try:
            for i,source in enumerate(files):
                if source.is_file():
                    target = trash/f'{i}{source.suffix}'
                    source.replace(target); moved.append((source,target))
            if model['kind']=='training':
                metadata = storage.root/'training'/model['run_id']/'run.json'
                record = read_json(metadata)
                record.update(model_deleted=True,model_deleted_at=datetime.now(timezone.utc).isoformat())
                write_json(metadata,record)
        except Exception:
            for source,target in reversed(moved):
                target.replace(source)
            raise
        for _,target in moved:
            target.unlink()
        trash.rmdir()
        return {'deleted':model_id,'files':len(moved)}
