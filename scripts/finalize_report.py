"""Append the documented execution amendment without changing statistical analysis."""
from pathlib import Path
import json

def main():
    out=Path(__file__).resolve().parents[1]/'reports/study-v1'
    verified=json.loads((out/'verification.json').read_text())
    assert verified['verified']
    p=out/'REPORT.md'; text=p.read_text(encoding='utf-8')
    marker='## Execution amendment and verification'
    text=text.replace('UCI source availability and dataset restrictions are documented in the audit, including pending access failures.', 'UCI source availability and dataset restrictions are documented in the audit; all initial source-access failures were resolved before training.')
    findings='## Research finding and timing limitations'
    if findings not in text:
        stats=json.loads((out/'statistics.json').read_text())
        timing=json.loads((out/'timing_audit.json').read_text())
        shaped=stats['summary']['shaped']; random=stats['summary']['random']
        text+='\n'+findings+'\n\n'
        matches=sum(p['same_configurations_and_scores'] for p in timing['shaped_vs_retrieval_pairs'])
        text+=f"The 30% improvement hypothesis is not supported. Shaped PPO changes the capped target-evaluation statistic by {1-shaped['n_target_capped']/random['n_target_capped']:.1%}, while outer macro-F1 falls from {random['outer_macro_f1']:.4f} for Random Search to {shaped['outer_macro_f1']:.4f}. Shaped and retrieval-only aggregate outer macro-F1, anytime score and capped target counts are equal. Complete configuration/score sequences match in {matches} of nine task/seed pairs. The added shaping term demonstrates no incremental measured benefit under this short-budget protocol. Cold fallback matches ordinary PPO exactly.\n\n"
        text+=f"The timing audit found {len(timing['elapsed_time_exceedances'])} trials whose elapsed time exceeded the configured 120-second worker response timeout. These observations remain in the raw results. Startup, send, scheduling and possible host suspension are not separately timed, so exact causes cannot be established. A strict end-to-end wall-time limit was not verified; wall-clock means must not be interpreted as algorithmic speed comparisons. See [timing audit](timing_audit.json).\n\n"
        text+='Some German Credit final refits also emitted logistic convergence warnings. Search-trial warning counts exclude final refit warnings; predictions were evaluated as returned, without solver retuning on test outcomes.\n\n'
        text+='Use `scripts/run_study.ps1` to verify saved evidence and regenerate this report. Use `python scripts/reproduce_study.py --scope full --output results/replication-001` for an independent isolated rerun after restoring audited data; see README.\n'
    if marker not in text:
        text+='\n'+marker+'\n\n'
        text+='After 27 Iris runs, German Credit failed at manifest parsing before any evaluation. Its categorical-column list had been serialized as a Python representation instead of JSON. The metadata encoding and one audit-builder line were corrected; processed data, splits, hyperparameters, selection, encoder, bank, evaluator, retrieval, and statistical analysis were unchanged. The original protocol and all 138 previously completed run records were archived. Resume signatures were migrated transparently, and results were reused without rerunning or selecting outcomes. See [protocol amendment](protocol_amendment.json) and [original evidence](amendment-001/).\n\n'
        text+=f"The completed package verifier checked {verified['total_evaluations']} evaluations, {verified['checked_episodes']} recorded episodes, all 16 dataset checksums, model/bank versions, exact budgets, split roles, reward telescoping, nine cold-start parity pairs, and preservation of the pre-amendment results. See [verification](verification.json).\n"
    p.write_text(text,encoding='utf-8')
    print('Execution amendment added to the report.')

if __name__=='__main__': main()
