"""Demonstrate the frozen transfer mechanism on a generated development task."""
import argparse
import gzip
import json
from pathlib import Path
import tempfile
import pandas as pd
from trajmeta.agents import run_ppo
from trajmeta.automl import Evaluator
from trajmeta.data import load_dataset, split_task
from trajmeta.retrieval import Retriever
from trajmeta.study import settings, file_hash

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--budget',type=int,default=4)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if not 1<=args.budget<=32: parser.error('Use a demo budget between 1 and 32.')
    root=Path(__file__).resolve().parents[1]
    config=settings(root)
    protocol=json.loads((root/'reports/study-v1/final_protocol.json').read_text())
    models=root/'models/study-v1'
    versions=json.loads((models/'versions.json').read_text())
    assert file_hash(models/'encoder.pt')==versions['encoder_sha256']
    manifest=pd.read_csv(root/'data/dataset_manifest.csv').set_index('dataset_key')
    task=split_task(*load_dataset(manifest.loc['synthetic_val'],root),seed=2026,folds=2)
    with tempfile.TemporaryDirectory(prefix='trajmeta-demo-') as directory:
        bank=Path(directory)/'bank.jsonl'
        bank.write_bytes(gzip.decompress((models/'bank.jsonl.gz').read_bytes()))
        assert file_hash(bank)==versions['bank_sha256']
        retriever=Retriever(bank,models/'encoder.pt',threshold=protocol['selection']['threshold'])
        evaluator=Evaluator(task.X_train,task.y_train,budget=args.budget,folds=2,seed=args.seed)
        try:
            episodes=run_ppo(evaluator,config['search_space'],args.seed,config['ppo'],retriever=retriever,strength=protocol['selection']['strength'],warm_mode='learned')
        finally:
            evaluator.close()
        result={'scope':'Synthetic development demonstration; not a new benchmark result',
          'charged_evaluations':len(evaluator.records),'best_inner_score':evaluator.best['inner_score'],
          'episodes':[{'configuration':e['pipeline_config'],'score':e['inner_score'],
             'steps':[{'action':s['action'],'base_reward':s['base_reward'],'shaped_reward':s['shaped_reward'],
                       'potential':s['potential'],'next_potential':s['next_potential'],
                       'retrieved':s['retrieval']['trajectory_ids']} for s in e['steps']]} for e in episodes]}
        print(json.dumps(result,indent=2))

if __name__=='__main__': main()
