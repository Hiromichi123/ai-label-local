from PIL import Image
from fastapi.testclient import TestClient
from app.storage import Storage


def test_browse_import_and_boundaries(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'nested').mkdir()
    Image.new('RGB', (40, 40)).save(source / 'one.jpg')
    (source / 'one.txt').write_text('0 0.5 0.5 0.2 0.2\n')
    Image.new('RGB', (40, 40)).save(source / 'nested' / 'two.png')
    original = (source / 'one.txt').read_bytes()
    storage = Storage(tmp_path / 'workspace')
    monkeypatch.setattr(api, 'storage', storage)
    monkeypatch.setenv('YOLO_LABEL_IMPORT_ROOTS', str(source))
    with TestClient(app) as client:
        roots = client.get('/api/server/directories').json()
        assert roots['directories'][0]['path'] == str(source.resolve())
        listing = client.get('/api/server/directories', params={'path': str(source)}).json()
        assert listing['image_count'] == 1
        assert listing['parent'] == ''
        assert listing['directories'][0]['name'] == 'nested'
        assert client.get('/api/server/directories', params={'path': str(tmp_path)}).status_code == 403
        assert client.get('/api/server/directories', params={'path': str(source / 'missing')}).status_code == 404
        body = {'name': 'remote', 'class_names': ['object'], 'path': str(source), 'recursive': False}
        result = client.post('/api/server/directories/import', json=body)
        assert result.status_code == 200, result.text
        assert result.json()['count'] == 1
        p = storage.project(result.json()['project_id'])
        row = next(iter(p['images'].values()))
        assert len(storage.annotations(p['id'], row['id'])['boxes']) == 1
        storage.save_annotations(p['id'], row['id'], 0, [])
        assert (source / 'one.txt').read_bytes() == original
        body['recursive'] = True
        assert client.post('/api/server/directories/import', json=body).json()['count'] == 2
        body['path'] = str(tmp_path)
        assert client.post('/api/server/directories/import', json=body).status_code == 403


def test_failed_import_removes_partial_project(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'bad.jpg').write_bytes(b'broken')
    storage = Storage(tmp_path / 'workspace')
    monkeypatch.setattr(api, 'storage', storage)
    monkeypatch.setenv('YOLO_LABEL_IMPORT_ROOTS', str(source))
    with TestClient(app) as client:
        response = client.post('/api/server/directories/import', json={'name': 'broken', 'class_names': ['object'], 'path': str(source)})
        assert response.status_code == 422
        assert storage.list_projects() == []


def test_invalid_label_fails_before_project_creation(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    source = tmp_path / 'source'
    source.mkdir()
    Image.new('RGB', (40, 40)).save(source / 'one.jpg')
    (source / 'one.txt').write_text('3 0.5 0.5 0.2 0.2\n')
    storage = Storage(tmp_path / 'workspace')
    monkeypatch.setattr(api, 'storage', storage)
    monkeypatch.setenv('YOLO_LABEL_IMPORT_ROOTS', str(source))
    with TestClient(app) as client:
        response = client.post('/api/server/directories/import', json={'name': 'invalid', 'class_names': ['object'], 'path': str(source)})
        assert response.status_code == 422
        assert 'one.jpg' in response.text
        assert storage.list_projects() == []
