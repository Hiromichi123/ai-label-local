import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.schemas import ProjectCreate, Box, TrainingRequest
from app.storage import Storage
from app.training import TrainingManager, snapshot, write_json, read_json


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    s = Storage(tmp_path/'workspace')
    p = s.create_project(ProjectCreate(name='custom',class_names=['one','two'],class_ids=[5,12]))
    files = []
    for i in range(3):
        f=tmp_path/f'{i}.jpg';Image.new('RGB',(64,64)).save(f);files.append(f)
    images=s.import_files(p['id'],files)
    for r in images:
        s.save_annotations(p['id'],r['id'],0,[Box(class_id=5,x1=4,y1=4,x2=20,y2=20),Box(class_id=12,x1=25,y1=25,x2=40,y2=40)])
    monkeypatch.setattr(api,'storage',s)
    manager=TrainingManager();monkeypatch.setattr(api,'training',manager)
    monkeypatch.delenv('YOLO_LABEL_DEFAULT_MODEL',raising=False)
    with TestClient(app) as c:
        yield c,s,p,images,manager


def test_custom_ids_migrate_and_reject_duplicates(setup):
    c,s,p,images,_=setup
    url=f"/api/projects/{p['id']}/classes"
    result=c.put(url+'/5',json={'name':'one','class_id':30})
    assert result.status_code==200,result.text
    assert result.json()['class_ids']==[30,12]
    assert s.annotations(p['id'],images[0]['id'])['boxes'][0]['class_id']==30
    assert s.annotations(p['id'],images[0]['id'])['version']==2
    assert c.put(url+'/30',json={'name':'one','class_id':12}).status_code==409
    assert c.post(url,json={'name':'three','class_id':100}).json()['class_id']==100
    assert c.post(url,json={'name':'four','class_id':100}).status_code==409
    assert list((s._project_path(p['id'])/'class_backups').glob('*/project.json'))
    assert c.put(f"/api/projects/{p['id']}/name",json={'name':'renamed'}).json()['name']=='renamed'


def test_class_migration_rolls_back(setup,monkeypatch):
    _,s,p,images,_=setup
    base=s._project_path(p['id'])
    files=[base/'project.json',*[base/'labels'/f"{r['id']}.txt" for r in images]]
    before=[f.read_bytes() for f in files]
    monkeypatch.setattr(s,'_write_json',lambda *_: (_ for _ in ()).throw(OSError('failed')))
    with pytest.raises(OSError):s.rename_class(p['id'],5,'one',30)
    assert [f.read_bytes() for f in files]==before


def test_sparse_training_snapshot_and_download(setup,tmp_path):
    c,s,p,images,_=setup
    target=tmp_path/'snapshot'
    snapshot(s,p['id'],target,TrainingRequest().model_dump())
    assert read_json(target/'class_mapping.json')['class_ids']==[5,12]
    for f in (target/'labels').glob('*/*.txt'):
        assert [int(line.split()[0]) for line in f.read_text().splitlines()]==[0,1]
    url=f"/api/projects/{p['id']}/images/{images[0]['id']}/download"
    response=c.get(url)
    assert response.status_code==200
    assert 'attachment;' in response.headers['content-disposition']
    with Image.open(io.BytesIO(response.content)) as image:assert image.size==(64,64)
    assert response.content==s.image_path(p['id'],images[0]['id']).read_bytes()


def test_models_default_protection_and_delete(setup,tmp_path):
    c,s,p,_,manager=setup
    model_root=tmp_path/'models';model_root.mkdir()
    old=model_root/'old.pt';old.write_bytes(b'old')
    new=model_root/'new.pt';new.write_bytes(b'new')
    models=c.get('/api/models').json()['models'];lookup={m['path']:m['id'] for m in models}
    assert c.put('/api/models/'+lookup[str(old)]+'/default').status_code==200
    assert c.delete('/api/models/'+lookup[str(old)]).status_code==409
    assert old.exists()
    c.put('/api/models/'+lookup[str(new)]+'/default')
    assert c.delete('/api/models/'+lookup[str(old)]).status_code==200
    assert not old.exists() and new.exists()
    assert c.delete('/api/models/not-a-model').status_code==404
    # Completed training model deletion removes checkpoints but keeps dataset and log.
    run_id='a'*32;directory=s.root/'training'/run_id
    weights=directory/'output'/'train'/'weights';weights.mkdir(parents=True)
    best=weights/'best.pt';best.write_bytes(b'best');(weights/'last.pt').write_bytes(b'last')
    (directory/'train.log').write_text('log');(directory/'dataset').mkdir()
    write_json(directory/'run.json',{'id':run_id,'project_id':p['id'],'name':'trained','status':'completed','model_path':str(best),'class_names':['one','two'],'params':{'imgsz':640},'created_at':'now'})
    entry=next(m for m in c.get('/api/models').json()['models'] if m.get('run_id')==run_id)
    result=c.delete('/api/models/'+entry['id'])
    assert result.status_code==200,result.text
    assert not best.exists() and not (weights/'last.pt').exists()
    assert (directory/'train.log').read_text()=='log'
    assert (directory/'dataset').is_dir()
    assert read_json(directory/'run.json')['model_deleted']
    assert not manager.view(s,p['id'],run_id)['model_available']


def test_prediction_maps_dense_model_ids_back_to_project_ids(setup,tmp_path,monkeypatch):
    import time
    from app import api
    from app.detectors.base import Detection
    c,s,p,images,_=setup
    model=tmp_path/'fake.pt';model.write_bytes(b'fake')
    class Detector:
        def predict(self,*_):return [Detection(0,.9,4,4,20,20),Detection(1,.8,25,25,40,40)]
    monkeypatch.setattr(api,'load_detector',lambda _:Detector())
    response=c.post(f"/api/projects/{p['id']}/predict",json={'scope':'current','image_id':images[0]['id'],'model_path':str(model)})
    assert response.status_code==200,response.text
    for _ in range(300):
        job=c.get('/api/jobs/'+response.json()['id']).json()
        if job['status'] in {'completed','failed'}:break
        time.sleep(.01)
    assert job['status']=='completed',job
    assert [b['class_id'] for b in s.annotations(p['id'],images[0]['id'])['boxes']]==[5,12]


def test_model_in_use_by_training_cannot_be_deleted(setup,tmp_path):
    c,s,p,_,manager=setup
    model_root=tmp_path/'models';model_root.mkdir()
    model=model_root/'training-base.pt';model.write_bytes(b'base')
    directory=s.root/'training'/('b'*32);directory.mkdir(parents=True)
    write_json(directory/'run.json',{'id':'b'*32,'project_id':p['id'],'status':'running','params':{'model_path':str(model)}})
    entry=next(m for m in c.get('/api/models').json()['models'] if m['path']==str(model))
    assert entry['delete_blocked']
    assert c.delete('/api/models/'+entry['id']).status_code==409
    assert model.read_bytes()==b'base'


def test_cross_project_model_uses_selected_class_for_unmatched_names(setup,tmp_path,monkeypatch):
    import time
    from app import api
    from app.detectors.base import Detection
    c,s,p,images,_=setup
    model=tmp_path/'foreign-model.pt';model.write_bytes(b'fake')
    class Detector:
        class_names=['foreign','one']
        def predict(self,*_):return [Detection(0,.9,4,4,20,20),Detection(1,.8,25,25,40,40)]
    monkeypatch.setattr(api,'load_detector',lambda _:Detector())
    url=f"/api/projects/{p['id']}/predict"
    response=c.post(url,json={'scope':'current','image_id':images[0]['id'],'model_path':str(model),'target_class_id':12})
    assert response.status_code==200,response.text
    for _ in range(300):
        job=c.get('/api/jobs/'+response.json()['id']).json()
        if job['status'] in {'completed','failed'}:break
        time.sleep(.01)
    assert job['status']=='completed',job
    assert [b['class_id'] for b in s.annotations(p['id'],images[0]['id'])['boxes']]==[12,5]
    label=s._project_path(p['id'])/'labels'/f"{images[0]['id']}.txt"
    before=label.read_bytes()
    bad=c.post(url,json={'scope':'current','image_id':images[0]['id'],'model_path':str(model),'target_class_id':99})
    assert bad.status_code==400
    assert label.read_bytes()==before
