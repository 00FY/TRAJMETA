import json
from hashlib import sha256
import pandas as pd
from trajmeta import study_data
from trajmeta.data import load_dataset

def test_preserved_audit_serializes_categorical_columns_as_json(tmp_path,monkeypatch):
    data=tmp_path/'data/processed/old.csv'; data.parent.mkdir(parents=True)
    data.write_text('category,target\n1,a\n2,b\n')
    report=dict(dataset_key='uci_1',source_id=1,source_url='https://example.invalid/one',licence='CC BY 4.0',access_date='2026-01-01',target_column='target',domain='fixture',duplicate_group='fixture',task_type='classification',path='data/processed/old.csv',sha256=sha256(data.read_bytes()).hexdigest(),verified=True,categorical_columns=['category'],raw_sha256='raw',metadata_sha256='metadata',content_fingerprint='content',doi='fixture',split='meta_train',name='Fixture')
    old=tmp_path/'data/audits/pilot-v1'; old.mkdir(parents=True)
    (old/'audit.json').write_text(json.dumps([report]))
    monkeypatch.setattr(study_data,'CANDIDATES',[1])
    output=tmp_path/'data/audits/study-v1'
    study_data.create(tmp_path,output)
    row=pd.read_csv(output/'dataset_manifest.csv').iloc[0]
    assert json.loads(row.categorical_columns)==['category']
    X,_=load_dataset(row,tmp_path)
    assert X.category.tolist()==['1','2']
