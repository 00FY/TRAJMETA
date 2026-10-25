"""Post-execution timing quality audit; no observations are removed."""
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]; out=root/'reports/study-v1'
    protocol=json.loads((out/'final_protocol.json').read_text()); limit=protocol['config']['trial_timeout_seconds']
    exceedances=[]
    for path in (root/'results/study-v1').glob('*/*/complete.json'):
        run=json.loads(path.read_text())
        for trial in run['records']:
            if trial['elapsed_seconds']>limit:
                exceedances.append({k:run[k] for k in ['phase','dataset_key','method','seed']}|{k:trial[k] for k in ['evaluation','elapsed_seconds','status']})
    rows=json.loads((out/'test_results.json').read_text()); pairs=[]
    for r in rows:
        if r['method']=='shaped':
            other=next(o for o in rows if o['method']=='retrieval' and o['dataset_key']==r['dataset_key'] and o['seed']==r['seed'])
            pairs.append({'dataset_key':r['dataset_key'],'seed':r['seed'],'same_configurations_and_scores':[(t['pipeline_config'],t['inner_score']) for t in r['records']]==[(t['pipeline_config'],t['inner_score']) for t in other['records']]})
    audit={'scope':'Post-execution measurement-quality diagnostic; no outcomes excluded or altered.', 'configured_worker_response_timeout_seconds':limit,'elapsed_time_exceedances':exceedances,'strict_end_to_end_wall_time_bound_verified':not bool(exceedances),'interpretation':'The worker polls after startup and send. Trial elapsed time covers more than that wait. Host scheduling or suspension is not separately measured. Exact causes of long stalls are unknown. Do not interpret wall-time means as algorithmic speed comparisons or claim a strict end-to-end 120-second bound.','shaped_vs_retrieval_pairs':pairs}
    (out/'timing_audit.json').write_text(json.dumps(audit,indent=2)); print(json.dumps(audit,indent=2))

if __name__=='__main__': main()
