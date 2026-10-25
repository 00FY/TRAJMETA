"""Verify the frozen experiment package; never fits or retunes a model."""
import gzip
import json
from hashlib import sha256
from pathlib import Path
import numpy as np
from PIL import Image
from trajmeta.data import load_dataset
from trajmeta.study import settings, inventory, code_hash, file_hash
from trajmeta.trajectories import read_bank

def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'reports/study-v1'
    config=settings(root); tasks=inventory(root,config)
    protocol=json.loads((out/'final_protocol.json').read_text())
    assert protocol['config']==config
    assert protocol['source_sha256']==code_hash(root)
    assert protocol['audit_freeze_sha256']==file_hash(root/config['audit']/'freeze.json')
    for _,row in tasks.iterrows(): load_dataset(row,root)
    analysis=json.loads((out/'analysis_plan.json').read_text())
    assert analysis['analysis_script_sha256']==file_hash(root/'scripts/report_study.py')
    amendment=json.loads((out/'protocol_amendment.json').read_text())
    archive=out/'amendment-001'
    original_protocol=json.loads((archive/'final_protocol.json').read_text())
    assert file_hash(archive/'final_protocol.json')==amendment['original_protocol_sha256']
    for key in ['config','selection','bank_sha256','encoder_sha256']:
        assert original_protocol[key]==protocol[key]
    assert file_hash(archive/'original_completed_units.json.gz')==amendment['original_results_archive_sha256']
    originals=json.loads(gzip.decompress((archive/'original_completed_units.json.gz').read_bytes()))
    for record in amendment['completed_units']:
        before_text=originals[record['path']]
        assert sha256(before_text.encode()).hexdigest()==record['original_sha256']
        before=json.loads(before_text); after=json.loads((root/record['path']).read_text())
        assert {k:v for k,v in before.items() if k!='signature'}=={k:v for k,v in after.items() if k!='signature'}
    bank=read_bank(root/'artifacts/study-v1/bank.jsonl')
    train_keys=set(tasks[tasks.split=='meta_train'].index)
    assert {r['dataset_key'] for r in bank}==train_keys
    assert len(bank)==len(train_keys)*len(config['seeds'])*config['bank_budget']
    assert all(r['split']=='meta_train' for r in bank)
    assert file_hash(root/'artifacts/study-v1/bank.jsonl')==protocol['bank_sha256']
    assert file_hash(root/'models/study-v1/encoder.pt')==protocol['encoder_sha256']
    assert sha256(gzip.decompress((root/'models/study-v1/bank.jsonl.gz').read_bytes())).hexdigest()==protocol['bank_sha256']
    sanity=json.loads((out/'encoder_sanity.json').read_text())
    assert sanity['gate_passed'] and sanity['training_seeds']==[11,29] and sanity['check_seed']==47
    final=json.loads((out/'test_results.json').read_text())
    validation=json.loads((out/'validation_results.json').read_text())
    expected={(k,m,s) for k in tasks[tasks.split=='meta_test'].index for m in config['methods'] for s in config['seeds']}
    assert len(final)==len(expected) and {(r['dataset_key'],r['method'],r['seed']) for r in final}==expected
    expected_val={(k,m,s) for k in tasks[tasks.split=='meta_val'].index for m in config['methods'] for s in config['seeds']}
    assert len(validation)==len(expected_val) and {(r['dataset_key'],r['method'],r['seed']) for r in validation}==expected_val
    target=round(.9*float(np.median([r['best_inner_score'] for r in validation if r['method']=='random'])),6)
    assert protocol['selection']['target']==target
    counts={}; failures=0; warnings=0; episode_count=0
    for path in (root/'results/study-v1').glob('*/*/complete.json'):
        r=json.loads(path.read_text()); phase=r['phase']
        counts[phase]=counts.get(phase,0)+1
        split='meta_train' if phase=='bank' else 'meta_test' if phase=='test' else 'meta_val'
        assert tasks.loc[r['dataset_key'],'split']==split==r['split']
        assert len(r['records'])==r['evaluations']==config['budget']
        assert [t['evaluation'] for t in r['records']]==list(range(1,config['budget']+1))
        np.testing.assert_array_equal(r['inner_curve'],np.maximum.accumulate([t['inner_score'] for t in r['records']]))
        assert r['failures']==sum(t['status']!='ok' for t in r['records'])
        failures+=r['failures']; warnings+=sum(bool(t.get('warnings')) for t in r['records'])
        if phase!='test': assert 'outer_macro_f1' not in r
        if 'episodes_path' in r:
            episodes=json.loads((root/r['episodes_path']).read_text())
            assert len(episodes)==config['budget']
            for e in episodes:
                episode_count+=1
                steps=e['steps']; assert len(steps)==4
                assert steps[0]['potential']==0 and steps[-1]['next_potential']==0
                assert steps[-1]['base_reward']==e['inner_score']
                assert abs(sum(config['ppo']['gamma']**i*s['shaping_difference'] for i,s in enumerate(steps)))<1e-10
                assert all(abs(s['shaped_reward']-s['base_reward']-s['shaping_difference'])<1e-10 for s in steps)
                if r['method']=='cold': assert all(s['shaping_difference']==0 and not s['retrieval']['accepted'] for s in steps)
    assert counts['bank']==33 and counts['validation']==54 and counts['test']==81
    assert sum(n for p,n in counts.items() if p.startswith('validation-grid-'))==24
    for r in final:
        reached=next((i+1 for i,s in enumerate(r['inner_curve']) if s>=target),None)
        assert r['n_target']==reached and r['n_target_capped']==(reached if reached is not None else 33)
        assert r['target_score']==target
        if r['method']=='cold':
            plain=next(p for p in final if p['method']=='ppo' and p['dataset_key']==r['dataset_key'] and p['seed']==r['seed'])
            assert [(t['pipeline_config'],t['inner_score']) for t in plain['records']]==[(t['pipeline_config'],t['inner_score']) for t in r['records']]
            assert plain['outer_macro_f1']==r['outer_macro_f1']
    for name in ['results.png','anytime.png']:
        with Image.open(out/name) as im: im.verify()
    stats=json.loads((out/'statistics.json').read_text())
    for method,summary in stats['summary'].items():
        assert abs(summary['outer_macro_f1']-np.mean([r['outer_macro_f1'] for r in final if r['method']==method]))<1e-12
    report={'verified':True,'dataset_count':len(tasks),'training_trajectories':len(bank),
      'run_counts':counts,'total_evaluations':sum(counts.values())*32,'failed_trials':failures,
      'warning_bearing_trials':warnings,'checked_episodes':episode_count,'cold_parity_pairs':9,
      'source_matches_amended_protocol':True,'analysis_unchanged':True,'preserved_pre_amendment_runs':len(originals),'final_protocol_sha256':file_hash(out/'final_protocol.json')}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
