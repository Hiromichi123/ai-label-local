from PIL import Image
from fastapi.testclient import TestClient
from app.schemas import ProjectCreate
from app.storage import Storage
from app.labels import validate_label_text


def test_all_invalid_lines_reported_with_original_numbers():
    text='5 0.5 0.5 0.2 0.2\n\nbad line\n12 nan 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n5 0 0 0.4 0.4\n5 0.5 0.5 0 0.2\n12 0.5 0.5 0.2 0.2\n'
    boxes,errors=validate_label_text(text,100,100,[5,12])
    assert len(boxes)==2
    assert [e['line'] for e in errors]==[3,4,5,6,7]
    assert all(e['message'] for e in errors)
    assert validate_label_text('',100,100,[5,12])==([],[])


def test_validation_endpoint_does_not_write_and_save_uses_same_errors(tmp_path,monkeypatch):
    from app import api
    from app.main import app
    s=Storage(tmp_path/'workspace')
    p=s.create_project(ProjectCreate(name='lines',class_names=['one'],class_ids=[5]))
    image=tmp_path/'image.jpg';Image.new('RGB',(100,100)).save(image)
    record=s.import_files(p['id'],[image])[0]
    monkeypatch.setattr(api,'storage',s)
    url=f"/api/projects/{p['id']}/images/{record['id']}/label-file"
    label=s._project_path(p['id'])/'labels'/f"{record['id']}.txt"
    before=label.read_bytes()
    request={'version':0,'text':'bad\n5 0.5 0.5 0.2 0.2\n5 1.2 0.5 0.2 0.2'}
    with TestClient(app) as c:
        result=c.post(url+'/validate',json=request)
        assert result.status_code==200
        assert not result.json()['valid']
        assert [e['line'] for e in result.json()['errors']]==[1,3]
        saved=c.put(url,json=request)
        assert saved.status_code==422
        assert saved.json()['detail']['errors']==result.json()['errors']
        assert label.read_bytes()==before
        request['text']='5 0.5 0.5 0.2 0.2\n'
        assert c.post(url+'/validate',json=request).json()=={'valid':True,'errors':[]}
        assert c.put(url,json=request).status_code==200
