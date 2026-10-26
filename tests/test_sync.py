import json
import subprocess
import sys
from pathlib import Path
import pytest
from scripts.sync_to_remote import package_project, RECEIVER
from test_crop import setup_image


def test_sync_rejects_pending_crops(tmp_path):
    storage, pid, _, _ = setup_image(tmp_path)
    with pytest.raises(ValueError):
        package_project(storage, pid, tmp_path / 'snapshot.tar.gz')


def test_sync_snapshot_receiver_and_backup(tmp_path):
    storage, pid, r, _ = setup_image(tmp_path)
    storage.crop_image(pid, r['id'], 0, 100, 0)
    archive = tmp_path / 'snapshot.tar.gz'
    package_project(storage, pid, archive)
    remote = tmp_path / 'remote'
    command = [sys.executable, '-c', RECEIVER, str(archive), str(remote), pid]
    first = subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(first.stdout)['images'] == 1
    assert (remote / 'projects' / pid / 'images' / (r['id'] + '.jpg')).read_bytes() == storage.image_path(pid, r['id']).read_bytes()
    settings_file = remote / 'projects' / pid / 'training_settings.json'
    settings_file.write_text('{"epochs":77}', encoding='utf-8')
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert json.loads(settings_file.read_text()) == {'epochs':77}
    assert len(list((remote / 'sync_backups').iterdir())) == 1
