from PIL import Image
from fastapi.testclient import TestClient
from app.storage import Storage
from app.schemas import ProjectCreate


def test_label_text_validation_and_version(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    storage = Storage(tmp_path / 'workspace')
    project = storage.create_project(ProjectCreate(name='editor', class_names=['object']))
    source = tmp_path / 'image.png'
    Image.new('RGB', (100, 100)).save(source)
    image = storage.import_files(project['id'], [source])[0]
    monkeypatch.setattr(api, 'storage', storage)
    url = f"/api/projects/{project['id']}/images/{image['id']}/label-file"
    with TestClient(app) as client:
        assert client.get(url).json()['text'] == ''
        for text in ['bad', '2 0.5 0.5 0.2 0.2', '0 nan 0.5 0.2 0.2', '0 0 0 0.5 0.5']:
            assert client.put(url, json={'version': 0, 'text': text}).status_code == 422
            assert client.get(url).json()['text'] == ''
        result = client.put(url, json={'version': 0, 'text': '0 0.5 0.5 0.2 0.4\n'})
        assert result.status_code == 200
        assert result.json()['boxes'][0]['x1'] == 40
        assert client.get(url).json()['text'] == '0 0.500000 0.500000 0.200000 0.400000\n'
        assert client.put(url, json={'version': 0, 'text': ''}).status_code == 409
        assert client.put(url, json={'version': 1, 'text': ''}).status_code == 200
        assert client.get(url).json()['text'] == ''
