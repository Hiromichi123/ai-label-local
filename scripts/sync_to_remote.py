"""Sync a saved local project: cropped images, labels and metadata travel together."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import tempfile
import uuid


RECEIVER = r'''
import sys,json,hashlib,tarfile,tempfile,shutil,re
from pathlib import Path
from datetime import datetime
archive,workspace,pid=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]
assert re.fullmatch(r'[a-zA-Z0-9_-]{8,64}',pid)
projects=workspace/'projects';projects.mkdir(parents=True,exist_ok=True)
stage=Path(tempfile.mkdtemp(prefix='.sync-',dir=projects))
with tarfile.open(archive,'r:gz') as package:
 for item in package.getmembers():
  path=stage/item.name
  if not item.isfile() or not path.resolve().is_relative_to(stage.resolve()):raise ValueError('invalid archive member')
  path.parent.mkdir(parents=True,exist_ok=True)
  with package.extractfile(item) as source,path.open('wb') as dest:shutil.copyfileobj(source,dest)
manifest=json.loads((stage/'_sync_manifest.json').read_text())
assert {p.relative_to(stage).as_posix() for p in stage.rglob('*') if p.is_file()}==set(manifest)|{'_sync_manifest.json'}
for name,digest in manifest.items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest,name
project=json.loads((stage/'project.json').read_text());assert project['id']==pid
for iid,r in project['images'].items():
 assert re.fullmatch(r'[a-zA-Z0-9_-]{8,64}',iid)
 assert not r.get('needs_crop') and r['width']==r['height']
 assert (stage/'images'/(iid+'.jpg')).is_file() and (stage/'labels'/(iid+'.txt')).is_file()
target=projects/pid;backup=None
if target.exists():
 if (target/'training_settings.json').is_file():shutil.copy2(target/'training_settings.json',stage/'training_settings.json')
 if (target/'originals').exists():shutil.copytree(target/'originals',stage/'originals')
 backups=workspace/'sync_backups';backups.mkdir(exist_ok=True)
 backup=backups/(pid+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'));target.rename(backup)
try:stage.rename(target)
except Exception:
 if backup is not None:backup.rename(target)
 raise
print(json.dumps({'project':pid,'images':len(project['images']),'remote_path':str(target),'backup':str(backup) if backup else None}))
'''


def package_project(storage, pid, archive):
    project = storage.project(pid)
    if any(r.get('needs_crop') or r.get('width') != r.get('height') for r in project['images'].values()):
        raise ValueError('请先确认全部非正方形图片的裁剪，再同步。')
    base = storage._project_path(pid)
    files = ['project.json']
    for iid in project['images']:
        storage.annotations(pid, iid)  # Validate saved YOLO labels before sending.
        files.extend([f'images/{iid}.jpg', f'labels/{iid}.txt'])
    # Freeze the snapshot; do not archive changing live files after computing hashes.
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary)
        manifest = {}
        for name in files:
            data = (base / name).read_bytes()
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        if storage.project(pid) != project:
            raise ValueError('标注在同步打包时发生变化，请保存后重试。')
        (stage / '_sync_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        with tarfile.open(archive, 'w:gz', compresslevel=1) as package:
            for name in [*files, '_sync_manifest.json']:
                package.add(stage / name, arcname=name)
    return project


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from app.storage import Storage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project')
    parser.add_argument('--host', default='hiro@jinjiao.net')
    parser.add_argument('--port', default=4022, type=int)
    parser.add_argument('--remote-workspace', default='/home/hiro/ai-label-local/workspace')
    args = parser.parse_args()
    storage = Storage()
    pid = args.project
    if not pid:
        projects = storage.list_projects()
        for i, p in enumerate(projects, 1):
            print(f'{i}. {p["name"]}  ({len(p["images"])} images)')
        if not projects:
            raise ValueError('当前本地工作区没有项目。')
        index = int(input('选择要同步的项目编号：')) - 1
        if index not in range(len(projects)):
            raise ValueError('项目编号无效。')
        pid = projects[index]['id']
    print('请先在网页保存当前标注；同步期间不要继续修改。裁剪后的工作图会与标签一起上传。')
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / 'project.tar.gz'
        project = package_project(storage, pid, archive)
        print(f'同步 {project["name"]}: {len(project["images"])} 张，{archive.stat().st_size / 1024**2:.1f} MiB。SSH 可能要求两次密码。')
        remote_archive = '/tmp/ai-label-sync-' + uuid.uuid4().hex + '.tar.gz'
        subprocess.run(['scp', '-P', str(args.port), str(archive), args.host + ':' + remote_archive], check=True)
        command = shlex.join(['python3', '-c', RECEIVER, remote_archive, args.remote_workspace, pid])
        subprocess.run(['ssh', '-p', str(args.port), args.host, command], check=True)
    print('同步完成。远端同 ID 项目已备份；其他项目不受影响。')


if __name__ == '__main__':
    main()
