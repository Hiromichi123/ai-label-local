import json
import time
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.schemas import ProjectCreate, Box, TrainingRequest
from app.storage import Storage
from app.training import TrainingManager, snapshot, read_json


@pytest.fixture
def training_setup(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    store = Storage(tmp_path/'workspace')
    project = store.create_project(ProjectCreate(name='train-test', class_names=['object']))
    sources = []
    for i in range(5):
        p = tmp_path/f'{i}.jpg'
        Image.new('RGB', (80, 60) if i == 4 else (64, 64)).save(p)
        sources.append(p)
    records = store.import_files(project['id'], sources)
    for r in records[:3]:
        store.save_annotations(project['id'], r['id'], 0, [Box(class_id=0,x1=10,y1=10,x2=40,y2=40)])
    p = store.project(project['id'])
    p['images'][records[3]['id']]['status'] = 'predicted'
    store._write_json(store._project_path(project['id'])/'project.json',p)
    model_root=tmp_path/'models'; model_root.mkdir()
    monkeypatch.setattr('app.training.MODEL_ROOT',model_root)
    model = model_root/'base.pt'
    model.write_bytes(b'test')
    monkeypatch.setattr(api,'storage',store)
    manager = TrainingManager()
    monkeypatch.setattr(api,'training',manager)
    with TestClient(app) as client:
        yield client,store,project['id'],records,model,manager


def test_snapshot_excludes_candidates_and_is_disjoint(training_setup, tmp_path):
    client,store,pid,records,model,manager = training_setup
    request = TrainingRequest(model_path=str(model), epochs=2)
    target = tmp_path/'snapshot'
    counts = snapshot(store,pid,target,request.model_dump())
    assert counts == {'train':2,'val':1}
    manifest = read_json(target/'manifest.json')
    assert {r['id'] for r in manifest} == {r['id'] for r in records[:3]}
    train = {p.stem for p in (target/'images'/'train').glob('*.jpg')}
    val = {p.stem for p in (target/'images'/'val').glob('*.jpg')}
    assert not train & val
    row = records[0]
    copied = next(target.glob(f'labels/*/{row["id"]}.txt'))
    original = copied.read_bytes()
    store.save_annotations(pid,row['id'],1,[])
    assert copied.read_bytes() == original
    assert client.get(f'/api/projects/{pid}/training').json()['summary']['reviewed'] == 3


def test_training_failure_logged_and_not_registered(training_setup, monkeypatch):
    client,store,pid,records,model,manager = training_setup
    class FakeProcess:
        def __init__(self, args, **kwargs):
            kwargs['stdout'].write(b'training failure for test\n')
        def wait(self): return 7
    monkeypatch.setattr('app.training.subprocess.Popen',FakeProcess)
    response = client.post(f'/api/projects/{pid}/training',json={'model_path':str(model),'epochs':2})
    assert response.status_code == 200,response.text
    run_id = response.json()['id']
    for _ in range(200):
        run = client.get(f'/api/projects/{pid}/training/{run_id}').json()
        if run['status'] not in {'running','preparing'}: break
        time.sleep(.01)
    assert run['status'] == 'failed'
    assert not run['model_available']
    assert 'training failure' in run['log']
    assert client.post(f'/api/projects/{pid}/training',json={'model_path':str(model),'imgsz':961}).status_code == 422
    assert client.get(f'/api/projects/{pid}/training/not-a-run').status_code == 404


def test_single_training_guard_and_restart_recovery(training_setup):
    client,store,pid,records,model,manager = training_setup
    manager.active = 'another-run'
    response = client.post(f'/api/projects/{pid}/training',json={'model_path':str(model)})
    assert response.status_code == 409
    from app.training import write_json
    run_id = 'a'*32
    directory = store.root/'training'/run_id
    directory.mkdir(parents=True)
    write_json(directory/'run.json',{'id':run_id,'project_id':pid,'status':'running','created_at':'now'})
    TrainingManager().recover(store)
    assert read_json(directory/'run.json')['status'] == 'failed'


def test_completed_model_available_with_metrics(training_setup, monkeypatch):
    from app.training import write_json
    client,store,pid,records,model,manager = training_setup
    class SuccessProcess:
        def __init__(self, args, **kwargs):
            config = read_json(Path(args[-1]))
            self.path = Path(config['path'])
            kwargs['stdout'].write(b'epoch 1 completed\n')
        def wait(self):
            best = self.path/'best.pt'
            best.write_bytes(b'checkpoint')
            write_json(self.path/'progress.json',{'epoch':1,'metrics':{'metrics/mAP50(B)':.5},'model_path':str(best)})
            return 0
    monkeypatch.setattr('app.training.subprocess.Popen',SuccessProcess)
    response = client.post(f'/api/projects/{pid}/training',json={'model_path':str(model),'epochs':1})
    run_id = response.json()['id']
    for _ in range(300):
        run = client.get(f'/api/projects/{pid}/training/{run_id}').json()
        if run['status'] not in {'running','preparing'}: break
        time.sleep(.01)
    assert run['status'] == 'completed'
    assert run['model_available']
    assert run['epoch'] == 1
    assert run['metrics']['metrics/mAP50(B)'] == .5
    assert run['class_names'] == ['object']
    assert client.get(f'/api/projects/{pid}/training').json()['runs'][0]['model_available']


def test_cancel_preparation(training_setup, monkeypatch):
    client,store,pid,records,model,manager = training_setup
    import threading
    entered = threading.Event()
    def slow_snapshot(store,pid,target,params,cancelled):
        entered.set()
        for _ in range(500):
            if cancelled(): raise InterruptedError('cancelled')
            time.sleep(.01)
        raise AssertionError('cancel not delivered')
    monkeypatch.setattr('app.training.snapshot',slow_snapshot)
    response = client.post(f'/api/projects/{pid}/training',json={'model_path':str(model)})
    run_id = response.json()['id']
    assert entered.wait(2)
    assert client.post(f'/api/projects/{pid}/training/{run_id}/stop').status_code == 200
    for _ in range(300):
        run = client.get(f'/api/projects/{pid}/training/{run_id}').json()
        if run['status'] == 'cancelled': break
        time.sleep(.01)
    assert run['status'] == 'cancelled'
    assert not run['model_available']


def test_worker_final_validation_does_not_add_epoch(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from app.training import write_json
    from app.train_worker import run
    class FakeYOLO:
        task = 'detect'
        def __init__(self, _): self.callbacks = {}
        def add_callback(self,name,fn): self.callbacks[name] = fn
        def train(self,**kwargs):
            for epoch in range(2):
                trainer = SimpleNamespace(epoch=epoch,metrics={'mAP':.5})
                self.callbacks['on_train_epoch_end'](trainer)
                self.callbacks['on_fit_epoch_end'](trainer)
            self.callbacks['on_fit_epoch_end'](SimpleNamespace(epoch=2,metrics={'mAP':.6}))
            best = Path(kwargs['project'])/kwargs['name']/'weights'/'best.pt'
            best.parent.mkdir(parents=True)
            best.write_bytes(b'checkpoint')
    monkeypatch.setitem(sys.modules,'ultralytics',SimpleNamespace(YOLO=FakeYOLO))
    params = TrainingRequest(epochs=2).model_dump()
    write_json(tmp_path/'config.json',{'path':str(tmp_path),'params':params})
    run(tmp_path/'config.json')
    progress = read_json(tmp_path/'progress.json')
    assert progress['epoch'] == 2
    assert progress['metrics']['mAP'] == .6
    assert Path(progress['model_path']).is_file()


def test_merged_snapshot_maps_names_and_namespaces_images(training_setup, tmp_path):
    client,store,pid,records,model,manager = training_setup
    other = store.create_project(ProjectCreate(name='second',class_names=['other','object'],class_ids=[0,9]))
    image = tmp_path/'other.jpg'
    Image.new('RGB',(64,64),'red').save(image)
    row = store.import_files(other['id'],[image])[0]
    store.save_annotations(other['id'],row['id'],0,[Box(class_id=0,x1=5,y1=5,x2=20,y2=20),Box(class_id=9,x1=25,y1=25,x2=45,y2=45)])
    # Different projects may contain the same local image id.
    directory=store._project_path(other['id'])
    other=store.project(other['id'])
    other['images'].pop(row['id'])
    for folder in ['images','labels']:
        suffix='.jpg' if folder=='images' else '.txt'
        (directory/folder/(row['id']+suffix)).rename(directory/folder/(records[0]['id']+suffix))
    row.update(id=records[0]['id'],status='edited',version=1)
    other['images'][row['id']]=row
    store._write_json(directory/'project.json',other)
    original=(directory/'labels'/f"{row['id']}.txt").read_bytes()
    params=TrainingRequest(project_ids=[pid,other['id'],pid],val_ratio=.25).model_dump()
    target=tmp_path/'merged'
    counts=snapshot(store,pid,target,params)
    assert sum(counts.values())==4
    manifest=read_json(target/'manifest.json')
    assert {entry['project_id'] for entry in manifest}=={pid,other['id']}
    assert len({entry['output_id'] for entry in manifest})==4
    labels=list(target.glob(f"labels/*/{other['id']}_{row['id']}.txt"))
    assert len(labels)==1
    assert [int(line.split()[0]) for line in labels[0].read_text().splitlines()]==[1,0]
    assert (directory/'labels'/f"{row['id']}.txt").read_bytes()==original
    assert read_json(target/'dataset.yaml')['names']==['object','other']
    mapping=read_json(target/'class_mapping.json')
    assert mapping['class_names']==['object','other']
    assert mapping['projects'][1]['mapping']=={'0':1,'9':0}
    train={f.stem for f in target.glob('images/train/*.jpg')}
    val={f.stem for f in target.glob('images/val/*.jpg')}
    assert not train & val


def test_training_weights_are_restricted_to_models(training_setup,tmp_path):
    client,store,pid,records,model,manager=training_setup
    outside=tmp_path/'outside.pt'; outside.write_bytes(b'outside')
    (model.parent/'ignore.onnx').write_bytes(b'not a training weight')
    nested=model.parent/'nested'; nested.mkdir()
    (nested/'second.PT').write_bytes(b'checkpoint')
    options=client.get('/api/training/options').json()
    assert {m['name'] for m in options['models']}=={'base.pt',str(Path('nested')/'second.PT')}
    assert options['root']==str(model.parent)
    assert options['projects'][0]['summary']['reviewed']==3
    response=client.post(f'/api/projects/{pid}/training',json={'model_path':str(outside)})
    assert response.status_code==422
    assert 'models' in response.json()['detail']['message']
    assert manager.active is None
    assert client.post(f'/api/projects/{pid}/training',json={'model_path':str(model),'project_ids':['notfound123']}).status_code==404


def test_multi_training_metadata_and_preferences(training_setup,monkeypatch):
    from app.training import write_json
    client,store,pid,records,model,manager=training_setup
    other=store.create_project(ProjectCreate(name='second',class_names=['different'],class_ids=[0]))
    images=store.import_files(other['id'],[store.image_path(pid,records[0]['id'])])
    store.save_annotations(other['id'],images[0]['id'],0,[Box(class_id=0,x1=5,y1=5,x2=25,y2=25)])
    class SuccessProcess:
        def __init__(self,args,**kwargs): self.path=Path(read_json(Path(args[-1]))['path'])
        def wait(self):
            best=self.path/'output'/'train'/'weights'/'best.pt'
            best.parent.mkdir(parents=True); best.write_bytes(b'checkpoint')
            write_json(self.path/'progress.json',{'epoch':1,'model_path':str(best)})
            return 0
    monkeypatch.setattr('app.training.subprocess.Popen',SuccessProcess)
    response=client.post(f'/api/projects/{pid}/training',json={'model_path':str(model),'project_ids':[pid,other['id']],'epochs':1})
    assert response.status_code==200,response.text
    run_id=response.json()['id']
    for _ in range(300):
        run=client.get(f'/api/projects/{pid}/training/{run_id}').json()
        if run['status'] not in {'preparing','running'}: break
        time.sleep(.01)
    assert run['status']=='completed',run
    assert run['class_names']==['object','different']
    assert run['class_ids']==[0,1]
    assert run['counts']=={'train':3,'val':1}
    assert [p['id'] for p in run['source_projects']]==[pid,other['id']]
    saved=client.get(f'/api/projects/{pid}/training-settings').json()['params']
    assert saved['project_ids']==[pid,other['id']]
    catalog=client.get('/api/models').json()['models']
    final=next(m for m in catalog if m.get('run_id')==run_id)
    assert final['class_names']==['object','different']


def test_training_metadata_retries_transient_permission_error(tmp_path,monkeypatch):
    from app.training import write_json
    path=tmp_path/'run.json'
    write_json(path,{'status':'running'})
    original=Path.read_text
    calls=[]
    def temporarily_busy(self,*args,**kwargs):
        if self==path and not calls:
            calls.append(True)
            raise PermissionError('file is being replaced')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',temporarily_busy)
    assert read_json(path)=={'status':'running'}
