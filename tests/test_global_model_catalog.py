import pytest
from app.labels import model_class_mapping
from app.schemas import ProjectCreate
from app.storage import Storage
from app.training import TrainingManager, write_json
from app.model_management import list_models


def test_model_classes_map_by_name_not_order_or_zero_based_id():
    project={'class_names':['cable','insulator','transformer'],'class_ids':[5,12,30]}
    assert model_class_mapping(['transformer','cable'],project)=={0:30,1:5}
    assert model_class_mapping(['insulator'],project)=={0:12}
    assert model_class_mapping([],project)=={0:5,1:12,2:30}
    with pytest.raises(ValueError,match='unknown'):
        model_class_mapping(['unknown'],project)


def test_global_model_catalog_includes_other_projects_and_excludes_deleted(tmp_path):
    s=Storage(tmp_path/'workspace')
    projects=[s.create_project(ProjectCreate(name=name,class_names=[name])) for name in ['cable','insulator']]
    for index,p in enumerate(projects):
        run_id=str(index+1)*32
        directory=s.root/'training'/run_id
        weights=directory/'output'/'train'/'weights';weights.mkdir(parents=True)
        best=weights/'best.pt';best.write_bytes(b'checkpoint')
        write_json(directory/'run.json',{'id':run_id,'project_id':p['id'],'name':'same-name','status':'completed','model_path':str(best),'class_names':p['class_names'],'params':{'imgsz':960,'epochs':80},'created_at':'2026-10-26'})
    manager=TrainingManager()
    catalog=list_models(s,manager)['models']
    assert {m['project_name'] for m in catalog}=={'cable','insulator'}
    assert len(catalog)==2
    metadata=s.root/'training'/('1'*32)/'run.json'
    from app.training import read_json
    record=read_json(metadata);record['model_deleted']=True;write_json(metadata,record)
    assert [m['project_name'] for m in list_models(s,manager)['models']]==['insulator']
