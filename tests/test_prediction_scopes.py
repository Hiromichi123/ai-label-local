import time
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.storage import Storage
from app.schemas import ProjectCreate, Box
from app.detectors.base import Detection


@pytest.fixture
def prediction_setup(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    s = Storage(tmp_path / 'workspace')
    p = s.create_project(ProjectCreate(name='scopes',class_names=['object']))
    files = []
    for i in range(4):
        path = tmp_path / f'{i}.png'
        Image.new('RGB', (90,60) if i == 3 else (64,64)).save(path)
        files.append(path)
    images = s.import_files(p['id'], files)
    human = [Box(class_id=0,x1=1,y1=1,x2=10,y2=10)]
    for r in images[:2]:
        s.save_annotations(p['id'],r['id'],0,human)
    project = s.project(p['id'])
    project['images'][images[1]['id']]['status'] = 'predicted'
    s._write_json(s._project_path(p['id'])/'project.json',project)
    model = tmp_path/'fake.pt'
    model.write_bytes(b'test')
    monkeypatch.setenv('YOLO_LABEL_DEFAULT_MODEL', str(model))
    monkeypatch.setattr(api,'storage',s)
    class Fake:
        def predict(self,*args):
            time.sleep(.02)
            return [Detection(0,.8,15,15,40,40)]
    monkeypatch.setattr(api,'load_detector',lambda _:Fake())
    with TestClient(app) as client:
        yield client,s,p['id'],images,Fake


def wait_job(client, response):
    assert response.status_code == 200,response.text
    job = response.json()
    for _ in range(300):
        job = client.get('/api/jobs/'+job['id']).json()
        assert 0 <= job['completed'] <= job['total']
        if job['status'] in ['completed','failed','cancelled']:
            return job
        time.sleep(.01)
    raise AssertionError('job timeout')


def test_current_overwrites_completed_and_backs_up(prediction_setup):
    c,s,p,images,_ = prediction_setup
    iid = images[0]['id']
    before = s.annotations(p,iid)
    job = wait_job(c,c.post(f'/api/projects/{p}/predict',json={'scope':'current','image_id':iid}))
    assert job['status']=='completed' and job['total']==1
    after = s.annotations(p,iid)
    assert after['boxes'] != before['boxes'] and after['version']==2
    assert s.image_record(p,iid)['status']=='predicted'
    backups = list((s._project_path(p)/'prediction_backups').glob('*/*.txt'))
    assert len(backups)==1 and '0.085938' in backups[0].read_text()


def test_unfinished_includes_drafts_excludes_completed_and_pending_crop(prediction_setup):
    c,s,p,images,_ = prediction_setup
    first = s.annotations(p,images[0]['id'])
    job=wait_job(c,c.post(f'/api/projects/{p}/predict',json={'scope':'unfinished'}))
    assert job['status']=='completed' and job['completed']==job['total']==2
    assert s.annotations(p,images[0]['id'])==first
    assert s.image_record(p,images[1]['id'])['version']==2
    assert s.image_record(p,images[2]['id'])['version']==1
    assert s.image_record(p,images[3]['id'])['version']==0


def test_current_errors_for_missing_or_uncropped(prediction_setup):
    c,s,p,images,_ = prediction_setup
    assert c.post(f'/api/projects/{p}/predict',json={'scope':'current'}).status_code==404
    assert c.post(f'/api/projects/{p}/predict',json={'scope':'current','image_id':images[3]['id']}).status_code==409


def test_failed_job_reports_processed_count_and_keeps_label(prediction_setup,monkeypatch):
    c,s,p,images,Fake=prediction_setup
    iid=images[0]['id'];before=s.annotations(p,iid)
    def fail(*args):raise RuntimeError('test inference failure')
    monkeypatch.setattr(Fake,'predict',fail)
    job=wait_job(c,c.post(f'/api/projects/{p}/predict',json={'scope':'current','image_id':iid}))
    assert job['status']=='failed' and job['completed']==job['total']==1
    assert 'test inference failure' in job['errors'][0]
    assert s.annotations(p,iid)==before


def test_human_edit_during_inference_is_not_overwritten(prediction_setup,monkeypatch):
    c,s,p,images,Fake=prediction_setup
    iid=images[0]['id']
    def edit(*args):
        s.save_annotations(p,iid,1,[Box(class_id=0,x1=2,y1=2,x2=12,y2=12)])
        return [Detection(0,.8,15,15,40,40)]
    monkeypatch.setattr(Fake,'predict',edit)
    job=wait_job(c,c.post(f'/api/projects/{p}/predict',json={'scope':'current','image_id':iid}))
    assert job['status']=='failed' and '未覆盖' in job['errors'][0]
    assert s.image_record(p,iid)['status']=='edited'
    assert s.annotations(p,iid)['boxes'][0]['x1']==pytest.approx(2,abs=.001)


def test_empty_human_review_persists_and_is_excluded(prediction_setup):
    c,s,p,images,_ = prediction_setup
    iid = images[2]['id']
    response=c.put(f'/api/projects/{p}/images/{iid}/annotations',json={'version':0,'boxes':[]})
    assert response.status_code==200 and response.json()['version']==1
    reloaded=Storage(s.root).image_record(p,iid)
    assert reloaded['version']==1 and reloaded['status']=='edited'
    job=wait_job(c,c.post(f'/api/projects/{p}/predict',json={'scope':'unfinished'}))
    assert job['total']==1
    assert s.image_record(p,iid)['status']=='edited'
