from __future__ import annotations

import importlib.util
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from .labels import load_label, serialize, class_ids

ACTIVE = {'preparing', 'running', 'stopping'}
MODEL_ROOT = Path(__file__).resolve().parent.parent/'models'

def initial_weights():
    root = MODEL_ROOT.resolve()
    files = []
    for candidate in root.rglob('*') if root.is_dir() else []:
        if candidate.is_symlink() or not candidate.is_file() or candidate.suffix.lower() != '.pt':
            continue
        path = candidate.resolve()
        if path.is_relative_to(root):
            files.append({'name':str(path.relative_to(root)), 'path':str(path)})
    return {'root':str(root), 'models':sorted(files, key=lambda item:item['name'].casefold())}

def training_sources(storage, project_id, params):
    ids = list(dict.fromkeys(params.get('project_ids') or [project_id]))
    projects = [storage.project(pid) for pid in ids]
    names = list(dict.fromkeys(name for project in projects for name in project['class_names']))
    mappings = {project['id']:{cid:names.index(name) for cid,name in zip(class_ids(project),project['class_names'])} for project in projects}
    return projects, names, mappings


def write_json(path: Path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    for attempt in range(50):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(.02)


def read_json(path: Path, default=None):
    for attempt in range(5):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except PermissionError:
            # Windows can briefly deny readers while an atomic update replaces the file.
            if attempt == 4:
                return default
            time.sleep(.02)
        except (OSError, ValueError):
            return default


def eligible_records(project):
    return [r for r in project['images'].values() if not r.get('needs_crop') and r.get('status') in {'approved', 'edited', 'labeled'}]


def training_summary(storage, project_id):
    project = storage.project(project_id)
    records = eligible_records(project)
    return {'total': len(project['images']), 'reviewed': len(records), 'excluded': len(project['images'])-len(records), 'class_names': project['class_names'], 'class_ids': class_ids(project)}


def snapshot(storage, project_id, target, params, cancelled=lambda: False):
    projects, names, mappings = training_sources(storage, project_id, params)
    records = [(project, record) for project in projects for record in eligible_records(project)]
    if len(records) < 2:
        raise ValueError('至少需要 2 张已审核图片，才能分别建立训练集和验证集')
    random.Random(params['seed']).shuffle(records)
    val_count = max(1, min(len(records)-1, round(len(records)*params['val_ratio'])))
    splits = {'val': records[:val_count], 'train': records[val_count:]}
    manifest = []
    positive = 0
    train_positive = 0
    for split, rows in splits.items():
        (target/'images'/split).mkdir(parents=True)
        (target/'labels'/split).mkdir(parents=True)
        for project, record in rows:
            source_project_id = project['id']
            if cancelled():
                raise InterruptedError('训练已停止')
            image_id = record['id']
            source = storage.image_path(source_project_id, image_id)
            label = storage._project_path(source_project_id)/'labels'/f'{image_id}.txt'
            if not label.is_file():
                raise ValueError(f"{record['filename']}: 标签文件缺失")
            before_image, before_label = source.stat(), label.stat()
            boxes = load_label(label, record['width'], record['height'], class_ids(project))
            dense = mappings[source_project_id]
            boxes = [box.model_copy(update={'class_id':dense[box.class_id]}) for box in boxes]
            positive += len(boxes)
            if split == 'train':
                train_positive += len(boxes)
            with Image.open(source) as image:
                if image.size != (record['width'], record['height']):
                    raise ValueError('图片尺寸与项目索引不一致')
            output_id = f'{source_project_id}_{image_id}' if len(projects)>1 else image_id
            shutil.copy2(source, target/'images'/split/f'{output_id}.jpg')
            (target/'labels'/split/f'{output_id}.txt').write_text(serialize(boxes, record['width'], record['height']), encoding='utf-8')
            after = storage.image_record(source_project_id, image_id)
            if (after.get('version') != record.get('version') or source.stat().st_mtime_ns != before_image.st_mtime_ns or label.stat().st_mtime_ns != before_label.st_mtime_ns):
                raise ValueError('建立快照时标注发生变化，请重试')
            manifest.append({'project_id':source_project_id, 'output_id':output_id, 'id': image_id, 'filename': record['filename'], 'version': record.get('version'), 'split': split})
    if not positive:
        raise ValueError('已审核图片中没有目标框，无法训练检测模型')
    if not train_positive:
        raise ValueError('当前划分的训练集没有目标框，请增加标注或调整验证比例/种子')
    for project in projects:
        current = storage.project(project['id'])
        if current['class_names'] != project['class_names'] or class_ids(current) != class_ids(project):
            raise ValueError('建立快照时类别发生变化，请重试')
    # JSON is valid YAML and preserves Unicode class names and absolute paths.
    write_json(target/'dataset.yaml', {'path': str(target.resolve()), 'train': 'images/train', 'val': 'images/val', 'names': names})
    write_json(target/'manifest.json', manifest)
    write_json(target/'class_mapping.json', {'class_ids':class_ids(projects[0]) if len(projects)==1 else list(range(len(names))), 'class_names':names, 'projects':[{'id':p['id'],'name':p['name'],'class_ids':class_ids(p),'class_names':p['class_names'],'mapping':mappings[p['id']]} for p in projects]})
    return {k: len(v) for k, v in splits.items()}


class TrainingManager:
    def __init__(self):
        self.lock = threading.RLock()
        self.active = None
        self.cancel = threading.Event()
        self.process = None

    def recover(self, storage):
        for path in (storage.root/'training').glob('*/run.json'):
            record = read_json(path, {})
            if record.get('status') in ACTIVE:
                record.update(status='failed', error='服务已重启，训练中断；可重新开始训练')
                write_json(path, record)

    def directory(self, storage, project_id, run_id):
        if not re.fullmatch(r'[0-9a-f]{32}', run_id):
            raise ValueError('invalid run id')
        path = storage.root/'training'/run_id
        record = read_json(path/'run.json')
        if not record or record['project_id'] != project_id:
            raise FileNotFoundError('training run not found')
        return path

    def view(self, storage, project_id, run_id, logs=False):
        path = self.directory(storage, project_id, run_id)
        record = read_json(path/'run.json', {})
        progress = read_json(path/'progress.json', {})
        record.update({k: v for k, v in progress.items() if k in {'epoch', 'metrics'}})
        record['epoch'] = min(record.get('epoch', 0), record.get('params', {}).get('epochs', record.get('epoch', 0)))
        record['model_available'] = record.get('status') == 'completed' and not record.get('model_deleted') and Path(record.get('model_path', '')).is_file()
        if logs:
            try:
                with (path/'train.log').open('rb') as stream:
                    stream.seek(0, 2)
                    stream.seek(max(0, stream.tell()-64000))
                    record['log'] = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', stream.read().decode('utf-8', errors='replace')).replace('\r', '\n')
            except OSError:
                record['log'] = ''
        return record

    def list(self, storage, project_id):
        runs = []
        for path in (storage.root/'training').glob('*/run.json'):
            r = read_json(path, {})
            if r.get('project_id') == project_id:
                runs.append(self.view(storage, project_id, r['id']))
        return sorted(runs, key=lambda r: r['created_at'], reverse=True)

    def start(self, storage, project_id, request):
        if importlib.util.find_spec('ultralytics') is None:
            raise ImportError('训练环境未安装，请安装项目的 pt 扩展')
        storage.project(project_id)
        if 'project_ids' in request.model_fields_set and not request.project_ids:
            raise ValueError('请至少勾选一个训练数据集')
        projects, names, mappings = training_sources(storage, project_id, request.model_dump())
        if sum(len(eligible_records(project)) for project in projects) < 2:
            raise ValueError('至少需要 2 张已审核图片')
        model = Path(request.model_path or os.environ.get('YOLO_LABEL_DEFAULT_MODEL', '')).expanduser().resolve()
        if str(model) not in {entry['path'] for entry in initial_weights()['models']}:
            raise ValueError('初始权重只能选择程序项目 models 目录内的 .pt 文件')
        with self.lock:
            if self.active:
                raise RuntimeError('已有训练任务在运行，请等待完成或先停止')
            run_id = uuid.uuid4().hex
            path = storage.root/'training'/run_id
            path.mkdir(parents=True)
            params = request.model_dump()
            params['model_path'] = str(model)
            params['project_ids'] = [p['id'] for p in projects]
            record = {'id': run_id, 'project_id': project_id, 'name': request.name, 'created_at': datetime.now(timezone.utc).isoformat(), 'status': 'preparing', 'epoch': 0, 'metrics': {}, 'params': params, 'class_names': names, 'class_ids': class_ids(projects[0]) if len(projects)==1 else list(range(len(names))), 'source_projects':[{'id':p['id'],'name':p['name']} for p in projects]}
            write_json(path/'run.json', record)
            self.active = run_id
            self.cancel.clear()
            threading.Thread(target=self._run, args=(storage, project_id, path, record), daemon=True).start()
            return record

    def _run(self, storage, project_id, path, record):
        try:
            record['counts'] = snapshot(storage, project_id, path/'dataset', record['params'], self.cancel.is_set)
            mapping = read_json(path/'dataset'/'class_mapping.json')
            record.update(class_names=mapping['class_names'], class_ids=mapping['class_ids'])
            if self.cancel.is_set():
                raise InterruptedError('训练已停止')
            write_json(path/'config.json', {'params': record['params'], 'path': str(path.resolve())})
            record['status'] = 'running'
            write_json(path/'run.json', record)
            env = os.environ.copy()
            env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', WANDB_MODE='disabled')
            with (path/'train.log').open('wb') as log:
                with self.lock:
                    if self.cancel.is_set():
                        raise InterruptedError('训练已停止')
                    self.process = subprocess.Popen([sys.executable, '-u', '-m', 'app.train_worker', str(path/'config.json')], cwd=Path(__file__).resolve().parent.parent, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=os.name != 'nt')
                code = self.process.wait()
            progress = read_json(path/'progress.json', {})
            if self.cancel.is_set():
                record.update(status='cancelled', error='已停止训练')
            elif code == 0 and progress.get('model_path') and Path(progress['model_path']).is_file():
                record.update(status='completed', model_path=progress['model_path'])
            else:
                record.update(status='failed', error=progress.get('error') or f'训练进程退出（{code}），请查看日志')
        except Exception as exc:
            record.update(status='cancelled' if self.cancel.is_set() else 'failed', error=str(exc))
        finally:
            with self.lock:
                try:
                    write_json(path/'run.json', record)
                finally:
                    self.active = None
                    self.process = None

    def stop(self, storage, project_id, run_id):
        with self.lock:
            path = self.directory(storage, project_id, run_id)
            if self.active == run_id:
                self.cancel.set()
                if self.process and self.process.poll() is None:
                    if os.name == 'nt':
                        self.process.terminate()
                    else:
                        try:
                            os.killpg(self.process.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
            return self.view(storage, project_id, run_id, logs=True)
