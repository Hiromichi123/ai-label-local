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
    model = tmp_path/'base.pt'
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
