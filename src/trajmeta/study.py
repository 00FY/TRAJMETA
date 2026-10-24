"""Small locally prespecified transfer study, validation lock, then sealed final evaluation."""
from hashlib import sha256
import gzip
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import yaml

from .automl import Candidate, Evaluator, final_score
from .agents import run_ppo, run_random, run_optuna
from .cli import provenance
from .data import audit_manifests, load_dataset, split_task
from .encoder import train as train_encoder
from .retrieval import Retriever
from .trajectories import write_bank

METHODS=['random','optuna','ppo','metadata','retrieval','shaped','final_config','simple_retrieval','cold']


def dump(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def code_hash(root):
    h=sha256()
    for path in sorted((Path(root)/'src').rglob('*.py')):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def inventory(root,config):
    directory=root/config['audit']
    freeze=json.loads((directory/'freeze.json').read_text())
    for name,h in freeze['hashes'].items():
        if file_hash(directory/name)!=h:
            raise ValueError('Study audit changed after freezing')
    return audit_manifests(pd.read_csv(directory/'dataset_manifest.csv'),
                          pd.read_csv(directory/'split_manifest.csv')).set_index('dataset_key')


def settings(root):
    path=root/'configs/study.yaml'
    config=yaml.safe_load(path.read_text())
    if config['methods']!=METHODS or config['seeds']!=[11,29,47]:
        raise ValueError('Unexpected comparison design; create a new study version')
    return config


def run_unit(root,config,rows,key,seed,method,phase,retriever=None,strength=0.,threshold=None):
    if phase not in {'bank', 'validation', 'test'} and not phase.startswith('validation-grid-'):
        raise ValueError('Unknown study phase')
    if phase=='bank' and rows.loc[key,'split']!='meta_train':
        raise ValueError('Held-out task cannot enter bank generation')
    if phase.startswith('validation') and rows.loc[key,'split']!='meta_val':
        raise ValueError('Validation selected a nonvalidation task')
    if phase=='test' and rows.loc[key,'split']!='meta_test':
        raise ValueError('Final evaluation requires a reserved test task')
    location=root/'results/study-v1'/phase/f'{key}-{method}-{seed}'
    complete=location/'complete.json'
    identity={'source':code_hash(root),'config':config,'key':key,'seed':seed,'method':method,
              'phase':phase,'strength':strength,'threshold':threshold}
    identity['audit_freeze'] = file_hash(root/config['audit']/'freeze.json')
    if retriever is not None:
        identity['bank'] = file_hash(root/'artifacts/study-v1/bank.jsonl')
        identity['encoder'] = file_hash(root/'artifacts/study-v1/encoder/encoder.pt')
    signature=sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if complete.exists():
        result=json.loads(complete.read_text())
        if result['signature']!=signature:
            raise ValueError('Completed run source/config changed; refusing stale resume')
        return result
    if location.exists():
        raise RuntimeError(f'Interrupted unit requires inspection: {location}')
    location.mkdir(parents=True)
    X,y=load_dataset(rows.loc[key],root)
    task=split_task(X,y,config['split_seed'],.25,config['folds'])
    budget=config['bank_budget'] if phase=='bank' else config['budget']
    ev=Evaluator(task.X_train,task.y_train,budget,config['folds'],seed,
                 split_seed=config['split_seed'],trial_timeout_seconds=config['trial_timeout_seconds'],
                 log_path=location/'evaluations.jsonl')
    start=perf_counter()
    episodes=[]
    try:
        if method in ['random','optuna']:
            {'random':run_random,'optuna':run_optuna}[method](ev,config['search_space'],seed)
        else:
            warm={'metadata':'metadata','final_config':'final_config','retrieval':'learned',
                  'shaped':'learned','simple_retrieval':'simple','cold':'learned'}.get(method)
            episodes=run_ppo(ev,config['search_space'],seed,config['ppo'],location/'policy.zip',
                             retriever=retriever,strength=strength,warm_mode=warm)
    finally:
        ev.close()
    elapsed=perf_counter()-start
    best=ev.best
    result={'signature':signature,'dataset_key':key,'split':rows.loc[key,'split'],'seed':seed,
            'method':method,'phase':phase,'evaluations':len(ev.records),
            'search_seconds':elapsed,'best_inner_score':best['inner_score'] if best else 0.,
            'failures':sum(r['status']!='ok' for r in ev.records),'records':ev.records,
            'inner_curve':np.maximum.accumulate([r['inner_score'] for r in ev.records]).tolist()}
    if phase=='test' and best:
        # Final fit is performed only after the entire search is finished.
        result.update(final_score(Candidate(**best['pipeline_config']),task,seed))
    if episodes:
        bank_records=[]
        for i,episode in enumerate(episodes):
            bank_records.append({**episode,'schema_version':1,'trajectory_id':f'{key}:{seed}:{i+1}',
                'dataset_key':key,'split':rows.loc[key,'split'],'seed':seed,'policy_version':'ppo-study-v2',
                'provenance_hash':signature})
        dump(location/'episodes.json',bank_records)
        result['episodes_path']=(location/'episodes.json').relative_to(root).as_posix()
        result['accepted_retrieval_steps']=sum(s.get('retrieval',{}).get('accepted',False) for e in episodes for s in e['steps'])
    if len(ev.records)!=budget:
        raise RuntimeError('Method overspent or underspent its evaluation budget')
    dump(complete,result)
    print(f'{phase}: {key} {seed} {method} best={result["best_inner_score"]:.4f} failures={result["failures"]}',flush=True)
    return result


def bank(root,config,rows):
    location=root/'artifacts/study-v1'
    location.mkdir(parents=True,exist_ok=True)
    records=[]
    results=[]
    for key in rows[rows.split=='meta_train'].index:
        for seed in config['seeds']:
            r=run_unit(root,config,rows,key,seed,'ppo','bank')
            results.append({k:v for k,v in r.items() if k!='records'})
            records.extend(json.loads((root/r['episodes_path']).read_text()))
    path=location/'bank.jsonl'
    if not path.exists():
        write_bank(path,records)
    dump(root/'reports/study-v1/bank_runs.json',results)
    return path


def references(root,config,rows,threshold,mode='learned',cold=False):
    return Retriever(root/'artifacts/study-v1/bank.jsonl',root/'artifacts/study-v1/encoder/encoder.pt',
                     threshold=threshold,top_k=config['retrieval']['top_k'],
                     temperature=config['retrieval']['temperature'],mode=mode,force_cold=cold)


def validate(root,config,rows):
    results=[]
    # Select shaping settings on validation tasks using inner scores only.
    candidates=[]
    for threshold in config['retrieval']['thresholds']:
        for strength in config['shaping_strengths']:
            retrieval=references(root,config,rows,threshold)
            group=[]
            phase=f'validation-grid-{threshold}-{strength}'
            for key in rows[rows.split=='meta_val'].index:
                for seed in config['seeds']:
                    group.append(run_unit(root,config,rows,key,seed,'shaped',phase,retrieval,strength,threshold))
            candidates.append({'threshold':threshold,'strength':strength,
                               'mean_inner_auc':float(np.mean([np.mean(r['inner_curve']) for r in group]))})
    winner=max(candidates,key=lambda x:(x['mean_inner_auc'],-x['strength'],-x['threshold']))
    threshold,strength=winner['threshold'],winner['strength']
    retrieval=references(root,config,rows,threshold)
    simple=references(root,config,rows,config['retrieval']['simple_threshold'],mode='simple')
    cold=references(root,config,rows,threshold,cold=True)
    for key in rows[rows.split=='meta_val'].index:
        for seed in config['seeds']:
            for method in METHODS:
                ref=None if method in ['random','optuna','ppo'] else (cold if method=='cold' else simple if method=='simple_retrieval' else retrieval)
                results.append(run_unit(root,config,rows,key,seed,method,'validation',ref,
                                        strength if method=='shaped' else 0.,threshold if ref is not None else None))
    target=.9*float(np.median([r['best_inner_score'] for r in results if r['method']=='random']))
    target=float(np.round(target,6))
    dump(root/'reports/study-v1/validation_grid.json',candidates)
    dump(root/'reports/study-v1/validation_results.json',results)
    return {'threshold':threshold,'strength':strength,'target':target,'target_definition':
            '0.9 times median best inner macro-F1 of validation Random Search runs; fixed for every test method/task'}


def lock(root,config,choice):
    path=root/'reports/study-v1/final_protocol.json'
    payload={'config':config,'selection':choice,'source_sha256':code_hash(root),
             'bank_sha256':file_hash(root/'artifacts/study-v1/bank.jsonl'),
             'encoder_sha256':file_hash(root/'artifacts/study-v1/encoder/encoder.pt'),
             'audit_freeze_sha256':file_hash(root/config['audit']/'freeze.json')}
    if path.exists() and json.loads(path.read_text())!=payload:
        raise ValueError('Final protocol is already locked; no post-test retuning')
    if not path.exists():
        dump(path,payload)
    return payload


def final_test(root,config,rows,protocol):
    if config != protocol['config']:
        raise ValueError('Configuration changed after the final protocol lock')
    if file_hash(root/config['audit']/'freeze.json') != protocol['audit_freeze_sha256']:
        raise ValueError('Audit changed after the final protocol lock')
    if code_hash(root)!=protocol['source_sha256']:
        raise ValueError('Code changed after the final protocol lock')
    if file_hash(root/'artifacts/study-v1/encoder/encoder.pt')!=protocol['encoder_sha256']:
        raise ValueError('Encoder changed after the protocol lock')
    if file_hash(root/'artifacts/study-v1/bank.jsonl')!=protocol['bank_sha256']:
        raise ValueError('Bank changed after the protocol lock')
    choice=protocol['selection']
    retrieval=references(root,config,rows,choice['threshold'])
    simple=references(root,config,rows,config['retrieval']['simple_threshold'],mode='simple')
    cold=references(root,config,rows,choice['threshold'],cold=True)
    results=[]
    for key in rows[rows.split=='meta_test'].index:
        for seed in config['seeds']:
            for method in METHODS:
                ref=None if method in ['random','optuna','ppo'] else (cold if method=='cold' else simple if method=='simple_retrieval' else retrieval)
                result=run_unit(root,config,rows,key,seed,method,'test',ref,
                                choice['strength'] if method=='shaped' else 0.,choice['threshold'] if ref is not None else None)
                result['target_score']=choice['target']
                result['n_target']=next((i+1 for i,s in enumerate(result['inner_curve']) if s>=choice['target']),None)
                result['n_target_capped']=result['n_target'] if result['n_target'] is not None else config['budget']+1
                results.append(result)
    # Strong exact cold-start parity assertion, not just a mean-score comparison.
    for r in [v for v in results if v['method']=='ppo']:
        c=next(v for v in results if v['dataset_key']==r['dataset_key'] and v['seed']==r['seed'] and v['method']=='cold')
        a=[(t['pipeline_config'],t['inner_score']) for t in r['records']]
        b=[(t['pipeline_config'],t['inner_score']) for t in c['records']]
        if a!=b:
            raise RuntimeError('Forced cold-start failed to reproduce plain PPO')
    dump(root/'reports/study-v1/test_results.json',results)
    return results


def execute(root, phase="all"):
    root=Path(root).resolve()
    config=settings(root)
    rows=inventory(root,config)
    bank_path=bank(root,config,rows)
    encoder_path=root/'artifacts/study-v1/encoder'
    if not (encoder_path/'encoder.pt').exists():
        train_encoder(bank_path,encoder_path)
    sanity=json.loads((encoder_path/'sanity.json').read_text())
    dump(root/'reports/study-v1/encoder_sanity.json',sanity)
    if not sanity['gate_passed']:
        raise RuntimeError('Encoder diagnostic gate failed; inspect before transfer evaluation')
    if phase == 'prepare':
        return
    protocol_path=root/'reports/study-v1/final_protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
    else:
        choice=validate(root,config,rows)
        protocol=lock(root,config,choice)
    if phase == 'validation':
        return
    results=final_test(root,config,rows,protocol)
    model_dir=root/'models/study-v1'
    model_dir.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copy2(encoder_path/'encoder.pt',model_dir/'encoder.pt')
    with gzip.GzipFile(filename='',mode='wb',fileobj=(model_dir/'bank.jsonl.gz').open('wb'),mtime=0) as f:
        f.write(bank_path.read_bytes())
    dump(model_dir/'versions.json',{'bank_sha256':protocol['bank_sha256'],'encoder_sha256':protocol['encoder_sha256']})
    print(f'Study complete: {len(results)} held-out runs',flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--phase', choices=['prepare','validation','all'], default='all')
    args=p.parse_args()
    execute(args.root, args.phase)
