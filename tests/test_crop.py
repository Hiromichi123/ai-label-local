import json
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient
from app.storage import Storage
from app.schemas import ProjectCreate, Box


def setup_image(tmp_path, size=(400, 200)):
    source = tmp_path / 'source.png'
    Image.new('RGB', size, (200, 80, 30)).save(source)
    source.with_suffix('.txt').write_text('0 .5 .5 .5 .5\n0 .95 .5 .1 .5\n')
    storage = Storage(tmp_path / 'workspace')
    project = storage.create_project(ProjectCreate(name='test', class_names=['object']))
    record = storage.import_files(project['id'], [source])[0]
    return storage, project['id'], record, source


def test_crop_is_permanent_and_transforms_labels(tmp_path):
    s, pid, r, source = setup_image(tmp_path)
    assert r['needs_crop'] and (r['width'], r['height']) == (400, 200)
    assert Image.open(s.image_path(pid, r['id'])).size == (400, 200)
    before = source.read_bytes()
    result = s.crop_image(pid, r['id'], 0, 100, 0)
    assert Image.open(s.image_path(pid, r['id'])).size == (200, 200)
    assert result['image']['needs_crop'] is False
    assert result['version'] == 1 and len(result['boxes']) == 1
    assert result['boxes'][0]['x1'] == pytest.approx(0)
    assert result['boxes'][0]['x2'] == pytest.approx(200)
    assert source.read_bytes() == before
    assert len(list((s._project_path(pid) / 'crop_backups').iterdir())) == 1
    assert Storage(s.root).image_record(pid, r['id'])['width'] == 200
    assert Storage(s.root).annotations(pid, r['id'])['boxes'] == result['boxes']


def test_pending_crop_cannot_save_and_stale_crop_conflicts(tmp_path):
    s, pid, r, _ = setup_image(tmp_path)
    with pytest.raises(RuntimeError):
        s.save_annotations(pid, r['id'], 0, [])
    with pytest.raises(ValueError):
        s.crop_image(pid, r['id'], 0, 201, 0)
    with pytest.raises(RuntimeError):
        s.crop_image(pid, r['id'], 1, 0, 0)
    s.crop_image(pid, r['id'], 0, 0, 0)
    with pytest.raises(RuntimeError):
        s.crop_image(pid, r['id'], 0, 0, 0)


def test_portrait_and_square_import(tmp_path):
    s, pid, r, _ = setup_image(tmp_path, (200, 400))
    result = s.crop_image(pid, r['id'], 0, 0, 200)
    assert result['image']['crop']['top'] == 200
    source = tmp_path / 'square.png'
    Image.new('RGB', (120, 120)).save(source)
    square = s.import_files(pid, [source])[0]
    assert not square['needs_crop']
    s.save_annotations(pid, square['id'], 0, [])


def test_crop_failure_restores_files(tmp_path, monkeypatch):
    s, pid, r, _ = setup_image(tmp_path)
    image = s.image_path(pid, r['id'])
    label = s._project_path(pid) / 'labels' / (r['id'] + '.txt')
    project = s._project_path(pid) / 'project.json'
    originals = [p.read_bytes() for p in (image, label, project)]
    monkeypatch.setattr(s, '_write_json', lambda *a: (_ for _ in ()).throw(OSError('simulated')))
    with pytest.raises(OSError):
        s.crop_image(pid, r['id'], 0, 100, 0)
    assert [p.read_bytes() for p in (image, label, project)] == originals


def test_http_crop_and_prediction_gate(tmp_path, monkeypatch):
    from app import api
    from app.main import app
    s, pid, r, _ = setup_image(tmp_path)
    monkeypatch.setattr(api, 'storage', s)
    with TestClient(app) as client:
        response = client.post(f'/api/projects/{pid}/predict', json={'model_path': 'not-used.pt'})
        assert response.status_code == 409
        response = client.post(f'/api/projects/{pid}/images/{r["id"]}/crop', json={'version': 0, 'left': 100, 'top': 0})
        assert response.status_code == 200 and response.json()['image']['width'] == 200
        assert client.get(f'/api/projects/{pid}/images/{r["id"]}/asset').headers['cache-control'] == 'no-store'
