from pathlib import Path
import json
from fastapi.testclient import TestClient
from app.schemas import ProjectCreate, TrainingRequest
from app.storage import Storage
from app.training import TrainingManager, write_json


def test_training_settings_persist_and_are_project_scoped(tmp_path,monkeypatch):
    from app import api
    from app.main import app
    storage=Storage(tmp_path/'workspace')
    first=storage.create_project(ProjectCreate(name='first',class_names=['object']))
    second=storage.create_project(ProjectCreate(name='second',class_names=['object']))
    monkeypatch.setattr(api,'storage',storage)
    monkeypatch.setattr(api,'training',TrainingManager())
    url=f"/api/projects/{first['id']}/training-settings"
    params=TrainingRequest(epochs=123,batch=6,device='cpu',hsv_v=.55,model_path='/models/initial.pt',lr0=.002).model_dump()
    with TestClient(app) as client:
        assert client.get(url).json()['source']=='defaults'
        assert client.put(url,json=params).json()['params']==params
        assert client.get(url).json()['params']==params
        assert client.get(f"/api/projects/{second['id']}/training-settings").json()['params']['epochs']==100
        # A new Storage instance reads the saved file, not process memory.
        monkeypatch.setattr(api,'storage',Storage(storage.root))
        assert client.get(url).json()=={'params':params,'source':'saved'}
        assert client.put(url,json={**params,'imgsz':961}).status_code==422
        assert client.get(url).json()['params']==params
        assert client.put('/api/projects/missing12/training-settings',json=params).status_code==404
    assert json.loads((storage._project_path(first['id'])/'training_settings.json').read_text(encoding='utf-8'))==params


def test_training_settings_fall_back_to_last_run(tmp_path,monkeypatch):
    from app import api
    from app.main import app
    storage=Storage(tmp_path/'workspace')
    project=storage.create_project(ProjectCreate(name='history',class_names=['object']))
    params=TrainingRequest(epochs=80,batch=8,device='cpu').model_dump()
    directory=storage.root/'training'/('a'*32);directory.mkdir(parents=True)
    write_json(directory/'run.json',{'id':'a'*32,'project_id':project['id'],'params':params,'status':'completed','created_at':'2026-10-26','class_names':['object']})
    monkeypatch.setattr(api,'storage',storage)
    monkeypatch.setattr(api,'training',TrainingManager())
    with TestClient(app) as client:
        result=client.get(f"/api/projects/{project['id']}/training-settings").json()
        assert result=={'params':params,'source':'last_run'}

