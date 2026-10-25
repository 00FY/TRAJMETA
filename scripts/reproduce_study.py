"""Rerun the frozen study in an isolated output directory, preserving published evidence."""
import argparse
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys
from trajmeta.study import settings,inventory,code_hash,file_hash,bank,validate,lock,final_test,dump
from trajmeta.encoder import train

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=['final','full'],default='final')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    source=Path(__file__).resolve().parents[1]
    destination=a.output.resolve()
    if destination==source or destination in source.parents:
        raise ValueError('Use a separate new output directory')
    if destination.is_relative_to(source) and not destination.is_relative_to(source/'results'):
        raise ValueError('Inside the repository, replication outputs must stay under results/')
    original=json.loads((source/'reports/study-v1/final_protocol.json').read_text())
    assert code_hash(source)==original['source_sha256']
    config=settings(source)
    marker=destination/'replication.json'
    identity={'source_sha256':original['source_sha256'],'scope':a.scope,'original_protocol_sha256':file_hash(source/'reports/study-v1/final_protocol.json')}
    if destination.exists():
        if not a.resume or not marker.exists() or json.loads(marker.read_text())!=identity:
            raise FileExistsError('Use a new output directory, or --resume for a matching replication')
    else:
        destination.mkdir(parents=True)
        shutil.copytree(source/'src',destination/'src',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copytree(source/'scripts',destination/'scripts',ignore=shutil.ignore_patterns('__pycache__'))
        (destination/'configs').mkdir()
        shutil.copy2(source/'configs/study.yaml',destination/'configs/study.yaml')
        shutil.copytree(source/config['audit'],destination/config['audit'])
        rows=inventory(source,config)
        for _,row in rows.iterrows():
            target=destination/row.path; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/row.path,target)
        dump(marker,identity)
    assert code_hash(destination)==original['source_sha256']
    rows=inventory(destination,config)
    out=destination/'reports/study-v1'; out.mkdir(parents=True,exist_ok=True)
    if a.scope=='final':
        artifacts=destination/'artifacts/study-v1'; (artifacts/'encoder').mkdir(parents=True,exist_ok=True)
        frozen=gzip.decompress((source/'models/study-v1/bank.jsonl.gz').read_bytes())
        (artifacts/'bank.jsonl').write_bytes(frozen)
        shutil.copy2(source/'models/study-v1/encoder.pt',artifacts/'encoder/encoder.pt')
        assert file_hash(artifacts/'bank.jsonl')==original['bank_sha256']
        assert file_hash(artifacts/'encoder/encoder.pt')==original['encoder_sha256']
        for name in ['final_protocol.json','validation_results.json','encoder_sanity.json','analysis_plan.json']:
            shutil.copy2(source/'reports/study-v1'/name,out/name)
    if a.prepare_only:
        print(f'Prepared verified {a.scope} replication at {destination}; rerun with --resume without --prepare-only to execute.')
        return
    if a.scope=='full':
        bank_path=bank(destination,config,rows)
        encoder=destination/'artifacts/study-v1/encoder'
        if not (encoder/'encoder.pt').exists(): train(bank_path,encoder)
        sanity=json.loads((encoder/'sanity.json').read_text()); assert sanity['gate_passed']
        dump(out/'encoder_sanity.json',sanity)
        if not (out/'final_protocol.json').exists():
            choice=validate(destination,config,rows)
            if choice!=original['selection']:
                raise RuntimeError('Validation choices differ from the published study; inspect replication before test')
            lock(destination,config,choice)
    protocol=json.loads((out/'final_protocol.json').read_text())
    actual=final_test(destination,config,rows,protocol)
    expected=json.loads((source/'reports/study-v1/test_results.json').read_text())
    differences=[]
    for r in actual:
        e=next(e for e in expected if (e['dataset_key'],e['method'],e['seed'])==(r['dataset_key'],r['method'],r['seed']))
        if r['inner_curve']!=e['inner_curve'] or r['outer_macro_f1']!=e['outer_macro_f1']:
            differences.append({'dataset_key':r['dataset_key'],'method':r['method'],'seed':r['seed']})
    dump(out/'replication_comparison.json',{'scope':a.scope,'runs':len(actual),'score_differences':differences,'timing_comparison':'Wall time is expected to differ; not used for exact reproducibility.'})
    subprocess.run([sys.executable,str(destination/'scripts/report_study.py')],check=True)
    subprocess.run([sys.executable,str(destination/'scripts/export_results.py')],check=True)
    print(f'Replication complete. Score differences: {len(differences)}. Published evidence was not overwritten.')

if __name__=='__main__': main()
