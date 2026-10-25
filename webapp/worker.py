"""One isolated exploratory search; never writes to the frozen study."""
import gzip
import io
import json
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import pandas as pd
from trajmeta.agents import run_ppo,run_random,run_optuna
from trajmeta.automl import Evaluator
from trajmeta.data import load_dataset,split_task
from trajmeta.retrieval import Retriever
from trajmeta.study import settings,file_hash

ROOT=Path(__file__).resolve().parents[1]
def write(path,value):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(value,allow_nan=False),encoding='utf-8'); temp.replace(path)

def execute(folder):
    request=json.loads((folder/'request.json').read_text()); config=settings(ROOT)
    if request['dataset']=='upload':
        frame=pd.read_csv(io.StringIO(request['csv']))
        if not 20<=len(frame)<=5000 or not 2<=len(frame.columns)<=61: raise ValueError('CSV must contain 20–5,000 rows and at most 60 features plus the target.')
        target=request['target']
        if target not in frame or frame[target].isna().any(): raise ValueError('Target must be present with no missing labels.')
        frame=frame.drop_duplicates()
        if frame[target].nunique()>20 or frame[target].value_counts().min()<6: raise ValueError('Use classification labels with 2–20 classes and at least six rows per class.')
        X,y=frame.drop(columns=target),frame[target]
        if frame.groupby(list(X.columns),dropna=False)[target].nunique().gt(1).any(): raise ValueError('Identical feature rows have conflicting labels; grouped splitting is required.')
    else:
        if request['dataset']=='synthetic_val': manifest=pd.read_csv(ROOT/'data/dataset_manifest.csv').set_index('dataset_key')
        else:
            manifest=pd.read_csv(ROOT/'data/audits/study-v1/dataset_manifest.csv').set_index('dataset_key')
            splits=pd.read_csv(ROOT/'data/audits/study-v1/split_manifest.csv').set_index('dataset_key')
            if splits.loc[request['dataset'],'split']!='meta_train': raise ValueError('Held-out tasks cannot be used for exploratory searches.')
        X,y=load_dataset(manifest.loc[request['dataset']],ROOT)
    task=split_task(X,y,seed=2026,folds=2)
    evaluator=Evaluator(task.X_train,task.y_train,request['budget'],2,request['seed'],split_seed=2026,trial_timeout_seconds=120,log_path=folder/'evaluations.jsonl')
    started=perf_counter(); episodes=[]; method=request['method']
    try:
        with tempfile.TemporaryDirectory(prefix='trajmeta-ui-') as temporary:
            retriever=None; strength=0; warm=None
            if method not in {'random','optuna','ppo'}:
                models=ROOT/'models/study-v1'; versions=json.loads((models/'versions.json').read_text()); protocol=json.loads((ROOT/'reports/study-v1/final_protocol.json').read_text())
                bank=Path(temporary)/'bank.jsonl'; bank.write_bytes(gzip.decompress((models/'bank.jsonl.gz').read_bytes()))
                if file_hash(bank)!=versions['bank_sha256'] or file_hash(models/'encoder.pt')!=versions['encoder_sha256']: raise ValueError('Frozen model or trajectory bank checksum mismatch.')
                retriever=Retriever(bank,models/'encoder.pt',threshold=.5 if method=='simple_retrieval' else protocol['selection']['threshold'],mode='simple' if method=='simple_retrieval' else 'learned',force_cold=method=='cold')
                strength=protocol['selection']['strength'] if method=='shaped' else 0
                warm={'metadata':'metadata','final_config':'final_config','simple_retrieval':'simple'}.get(method,'learned')
            if method in {'random','optuna'}: {'random':run_random,'optuna':run_optuna}[method](evaluator,config['search_space'],request['seed'])
            else: episodes=run_ppo(evaluator,config['search_space'],request['seed'],config['ppo'],retriever=retriever,strength=strength,warm_mode=warm)
        write(folder/'result.json',{'scope':'Exploratory development search; separate from published study','best':evaluator.best,'records':evaluator.records,'episodes':episodes,'elapsed_seconds':perf_counter()-started,'training_rows':len(task.X_train),'features':X.shape[1]})
    finally: evaluator.close()

if __name__=='__main__':
    folder=Path(sys.argv[1])
    try: execute(folder)
    except Exception as exc:
        write(folder/'error.json',{'error':str(exc)}); raise
