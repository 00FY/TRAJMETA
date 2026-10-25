"""Close candidate decisions and freeze a modest CPU study without moving old holdouts."""
import json
from pathlib import Path
from datetime import datetime, timezone
import pandas as pd

from .benchmark import CANDIDATES, SELECTED, metadata, fetch, licence_evidence, prepare_frame, digest
from .data import audit_manifests, split_task


def create(root, output, retry_pending=False):
    root, output = Path(root), Path(output)
    cached = {}
    if (output/'freeze.json').exists():
        if not retry_pending or (root/'results/study-v1').exists():
            raise FileExistsError('Study already frozen; retry only before any study runs')
        journal = json.loads((output/'journal.json').read_text())
        cached = {d['source_id']: d for d in journal['decisions'] if d['status'] != 'admitted' and not any(t in d.get('reason','').lower() for t in ['timed out','urlopen','getaddrinfo','connection','http error'])}
        (output/'freeze.json').rename(output/'freeze-before-retry.json')
    output.mkdir(parents=True, exist_ok=True)
    old = json.loads((root / 'data/audits/pilot-v1/audit.json').read_text())
    old_by_id={r['source_id']:r for r in old}
    rows, splits, reports, decisions = [], [], [], []
    # Group/site identifiers unavailable: random rows cannot support this protocol.
    exclusions = {18: 'Pittsburgh Bridges has no single documented target in the canonical API schema',
                  50: 'Image-level grouping unavailable; pixel/region leakage cannot be excluded',
                  149: 'Malformed singleton target in canonical CSV; source correction review needed'}
    for dataset_id in CANDIDATES:
        decision = {'source_id': dataset_id, 'checked_at_utc': datetime.now(timezone.utc).isoformat()}
        if dataset_id in old_by_id:
            previous=old_by_id[dataset_id]
            encoded=(root/previous['path']).read_bytes()
            if digest(encoded)!=previous['sha256']:
                raise ValueError('Existing audited data changed')
            destination=root/'data/processed/study-v1'/f"uci_{dataset_id}.csv"
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(encoded)
            report={**previous,'path':destination.relative_to(root).as_posix()}
            row={k:report[k] for k in ['dataset_key','source_id','source_url','licence','access_date','target_column',
                 'domain','duplicate_group','task_type','path','sha256','verified','categorical_columns',
                 'raw_sha256','metadata_sha256','content_fingerprint','doi']}
            row['categorical_columns'] = json.dumps(report['categorical_columns'])
            rows.append(row)
            splits.append({'dataset_key':report['dataset_key'],'split':report['split']})
            reports.append(report)
            decisions.append({**decision,'name':report['name'],'source_url':report['source_url'],
                 'licence':report['licence'],'status':'admitted','split':report['split'],
                 'reason':'Preserved previous individually verified audit and unchanged checksum'})
            print(f'audit {dataset_id}: preserved previous audit',flush=True)
            continue
        if dataset_id in cached:
            decisions.append(cached[dataset_id])
            continue
        try:
            meta, meta_raw = metadata(dataset_id)
            evidence = licence_evidence(fetch(meta['repository_url']))
            decision.update(name=meta['name'], source_url=meta['repository_url'], **evidence)
            if dataset_id in exclusions:
                raise ValueError(exclusions[dataset_id])
            if 'Classification' not in meta['tasks']:
                raise ValueError('Not classification')
            if meta['num_instances'] > 5000 or (meta['num_features'] or 0) > 60:
                raise ValueError('Outside preregistered CPU limits: 5000 rows / 60 features')
            if meta['num_instances'] < 100:
                raise ValueError('Fewer than 100 observations')
            if not meta.get('data_url'):
                raise ValueError('No canonical machine-readable UCI CSV')
            raw = fetch(meta['data_url'])
            frame, report = prepare_frame(meta, raw)
            target = report['target_column']
            if frame[target].value_counts().min() < 6:
                raise ValueError('Minority class below six after deduplication')
            split_task(frame.drop(columns=target), frame[target], seed=2026, folds=2)
            key = f'uci_{dataset_id}'
            split, group, note = SELECTED.get(dataset_id, ('meta_train', f'uci_{dataset_id}',
                'Additional training task admitted by licence, size, schema and duplicate checks, before scores'))
            if dataset_id in [15, 17]:
                group = 'wisconsin_breast_cancer'
            encoded = frame.to_csv(index=False, lineterminator='\n').encode()
            destination = root / 'data/processed/study-v1' / f'{key}.csv'
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(encoded)
            row = {'dataset_key': key, 'source_id': dataset_id, 'source_url': meta['repository_url'],
                   'licence': evidence['licence'], 'access_date': decision['checked_at_utc'],
                   'target_column': target, 'domain': meta['area'], 'duplicate_group': group,
                   'task_type': 'classification', 'path': destination.relative_to(root).as_posix(),
                   'sha256': digest(encoded), 'verified': True,
                   'categorical_columns': json.dumps(report['categorical_columns']),
                   'raw_sha256': digest(raw), 'metadata_sha256': digest(meta_raw),
                   'content_fingerprint': report['content_fingerprint'], 'doi': meta['dataset_doi']}
            rows.append(row)
            splits.append({'dataset_key': key, 'split': split})
            reports.append({**row, **report, **evidence, 'name': meta['name'], 'review': note,
                            'creators': meta['creators'], 'data_url': meta['data_url'],
                            'source_last_updated': meta['last_updated'], 'split': split})
            decision.update(status='admitted', split=split, reason=note)
        except Exception as exc:
            pending = any(t in str(exc).lower() for t in ['timed out','urlopen','getaddrinfo','connection','http error'])
            decision.update(status='pending_source_access' if pending else 'excluded_from_study', reason=str(exc))
        decisions.append(decision)
        (output/'journal.json').write_text(json.dumps({'decisions':decisions,'reports':reports},indent=2))
        print(f"audit {dataset_id}: {decision['status']} {decision.get('reason','')}", flush=True)
    present = {r['source_id']: r for r in reports}
    for previous in old:
        if previous['source_id'] not in present:
            raise ValueError(f"Previously reserved dataset lost: {previous['dataset_key']}; inspect before proceeding")
        current = present[previous['source_id']]
        if current['sha256'] != previous['sha256'] or current['split'] != previous['split']:
            raise ValueError('An old dataset or split changed')
    manifest, split_frame = pd.DataFrame(rows), pd.DataFrame(splits)
    audit_manifests(manifest, split_frame)
    manifest.to_csv(output / 'dataset_manifest.csv', index=False)
    split_frame.to_csv(output / 'split_manifest.csv', index=False)
    pd.DataFrame(decisions).to_csv(output / 'candidate_decisions.csv', index=False)
    (output / 'audit.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    names = ['dataset_manifest.csv', 'split_manifest.csv', 'audit.json', 'candidate_decisions.csv']
    (output / 'freeze.json').write_text(json.dumps({'scope': 'small CPU study-v1',
        'pending_source_access': [d['source_id'] for d in decisions if d['status']=='pending_source_access'],
        'audit_complete': not any(d['status']=='pending_source_access' for d in decisions),
        'hashes': {n: digest((output/n).read_bytes()) for n in names},
        'preserved_holdouts': [r['dataset_key'] for r in old if r['split'] != 'meta_train'],
        'selection_basis': 'licence, size, role schema, duplicates and split feasibility; no added-task scores'}, indent=2))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retry-pending', action='store_true')
    args = parser.parse_args()
    create(args.root, args.output, args.retry_pending)
