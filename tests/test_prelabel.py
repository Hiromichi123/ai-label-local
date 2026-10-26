from PIL import Image
from fastapi.testclient import TestClient
from app.storage import Storage
from app.schemas import ProjectCreate, Box
from app.detectors.base import Detection
import time


def test_prediction_keeps_saved_labels(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    storage = Storage(tmp_path / 'workspace')
    project = storage.create_project(ProjectCreate(name='cables', class_names=['线缆小', '线缆大']))
    sources = []
    for i in range(2):
        p = tmp_path / f'{i}.png'
        Image.new('RGB', (64, 64)).save(p)
        sources.append(p)
    images = storage.import_files(project['id'], sources)
    storage.save_annotations(project['id'], images[0]['id'], 0, [Box(class_id=0,x1=2,y1=2,x2=20,y2=20)])
    label = storage._project_path(project['id']) / 'labels' / (images[0]['id'] + '.txt')
    before = label.read_bytes()
    model = tmp_path / 'model.pt'
    model.write_bytes(b'test only')
    class FakeDetector:
        def predict(self, *args):
            return [Detection(0, .9, 5, 5, 30, 30)]
    monkeypatch.setattr(api, 'storage', storage)
    monkeypatch.setattr(api, 'load_detector', lambda path: FakeDetector())
    monkeypatch.setenv('YOLO_LABEL_DEFAULT_MODEL', str(model))
    monkeypatch.setenv('YOLO_LABEL_DEFAULT_IMAGE_SIZE', '960')
    with TestClient(app) as client:
        config = client.get('/api/config').json()
        assert config['default_model_path'] == str(model) and config['default_image_size'] == 960
        response = client.post('/api/projects/' + project['id'] + '/predict', json={})
        assert response.status_code == 200
        job = response.json()
        for _ in range(100):
            job = client.get('/api/jobs/' + job['id']).json()
            if job['status'] in ('completed','failed'):
                break
            time.sleep(.01)
        assert job['status'] == 'completed' and job['total'] == 1
        assert label.read_bytes() == before
        assert len(storage.annotations(project['id'], images[1]['id'])['boxes']) == 1
