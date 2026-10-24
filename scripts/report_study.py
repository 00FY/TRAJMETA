"""Rebuild descriptive results without training or changing the frozen study code."""
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/study-v1'

def paired(rows,method,baseline,metric):
    tasks=sorted({r['dataset_key'] for r in rows})
    delta=[]
    for task in tasks:
        a={r['seed']:r[metric] for r in rows if r['dataset_key']==task and r['method']==method}
        b={r['seed']:r[metric] for r in rows if r['dataset_key']==task and r['method']==baseline}
        assert a.keys()==b.keys()
        delta.append(float(np.mean([a[s]-b[s] for s in a])))
    delta=np.asarray(delta)
    # Enumerate the small empirical task bootstrap exactly; seeds stay within tasks.
    boot=np.asarray([np.mean(delta[list(ids)]) for ids in itertools.product(range(len(tasks)),repeat=len(tasks))])
    flips=np.asarray([np.mean(delta*np.asarray(signs)) for signs in itertools.product([-1,1],repeat=len(tasks))])
    return {'task_differences':dict(zip(tasks,delta.tolist())),'mean_difference':float(delta.mean()),
            'task_bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'exact_two_sided_task_sign_flip_p':float(np.mean(np.abs(flips)>=abs(delta.mean())-1e-12))}

def main():
    rows=json.loads((OUT/'test_results.json').read_text())
    protocol=json.loads((OUT/'final_protocol.json').read_text())
    sanity=json.loads((OUT/'encoder_sanity.json').read_text())
    for r in rows:
        r['inner_anytime_auc']=float(np.mean(r['inner_curve']))
        r['target_reached']=int(r['n_target'] is not None)
    methods=protocol['config']['methods']
    metrics=['outer_macro_f1','outer_balanced_accuracy','inner_anytime_auc','n_target_capped','search_seconds','target_reached']
    summary={m:{k:float(np.mean([r[k] for r in rows if r['method']==m])) for k in metrics} for m in methods}
    comparisons={b:{k:paired(rows,'shaped',b,k) for k in ['outer_macro_f1','inner_anytime_auc','n_target_capped']} for b in ['random','ppo','retrieval','simple_retrieval','final_config']}
    trials=[t for r in rows for t in r['records']]
    warning_counts=Counter(w for t in trials for w in t.get('warnings',[]))
    result={'summary':summary,'paired_shaped_minus_baseline':comparisons,'test_runs':len(rows),
            'test_evaluations':len(trials),'failed_test_trials':sum(t['status']!='ok' for t in trials),
            'warning_bearing_test_trials':sum(bool(t.get('warnings')) for t in trials),
            'warnings':dict(warning_counts),'inference_unit':'dataset; seeds averaged within each dataset',
            'limitations':'Only three test datasets. Bootstrap intervals are descriptive and unstable; no multiple-comparison-adjusted confirmatory claim.'}
    (OUT/'statistics.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    baseline=summary['random']['n_target_capped']
    reduction=1-summary['shaped']['n_target_capped']/baseline
    lines=['# TrajMeta-AutoML: controlled study report','',
      'This is a small CPU research study. It implements and evaluates trajectory transfer; it does not establish broad AutoML superiority.', '',
      '## Design','',
      'Forty-eight classification pipelines, 32 charged evaluations per search, two fixed inner folds, and seeds 11, 29, 47. Repeated configurations and failures consume budget. All final methods use the same outer split. Outer labels are unavailable to search, retrieval, and tuning.', '',
      'The training-only bank supervises one shared LSTM with within-task quality ranks. Terminal outcomes are masked from input. Metadata shortlists three source tasks, then cosine similarity selects high-ranked trajectories. Offline behavior cloning transfers their configurations without target evaluations. Shaped PPO adds a bounded potential difference, with zero terminal potential; retrieval-only uses the same warm start with zero shaping strength.', '',
      'The encoder check holds out seed 47 within training tasks. It is not a dataset-disjoint generalization test. Metadata can encode dataset identity indirectly. Action aliases and prior best-score observations may affect representations. Validation tasks choose the gate/strength; final tasks are evaluated only after the source, bank, encoder, audit, and configuration are locked.', '',
      f"Selected gate: {protocol['selection']['threshold']}; shaping strength: {protocol['selection']['strength']}; fixed target macro-F1: {protocol['selection']['target']:.6f}.", '',
      '## Results','']
    for m in methods:
        v=summary[m]
        lines.append(f"- {m}: outer macro-F1 {v['outer_macro_f1']:.4f}; inner anytime mean {v['inner_anytime_auc']:.4f}; target reached {v['target_reached']:.1%}; capped target evaluations {v['n_target_capped']:.2f}; mean search time {v['search_seconds']:.2f}s.")
    lines.extend(['',f"Shaped versus random changes the capped target-evaluation statistic by {reduction:.1%} (positive means fewer). Unreached targets are assigned budget + 1 for this descriptive statistic; this is not an uncensored time-to-target estimate or proof of the proposed 30% improvement.", '',
      f"There were {len(rows)} final runs, {len(trials)} charged evaluations, {result['failed_test_trials']} failed trials, and {result['warning_bearing_test_trials']} warning-bearing trials. See statistics.json and test_results.json for all outcomes.", '',
      '## Paired uncertainty','',
      'Seeds are averaged within each dataset before inference. Exact task-level sign flips and an enumerated empirical task bootstrap are reported. With only three final datasets, uncertainty is substantial and the minimum two-sided sign-flip p-value is 0.25. No comparison can establish conventional statistical significance. Multiple ablations are exploratory.', ''])
    for b,c in comparisons.items():
        v=c['outer_macro_f1']; lo,hi=v['task_bootstrap_95_interval']
        lines.append(f"- Shaped minus {b}, outer macro-F1: {v['mean_difference']:+.4f}; descriptive interval [{lo:+.4f}, {hi:+.4f}]; exact p={v['exact_two_sided_task_sign_flip_p']:.3f}.")
    lines.extend(['','## Encoder diagnostics','',f"Quality AUC: {sanity['quality_auc']:.4f}; simple kNN AUC: {sanity['simple_knn_quality_auc']:.4f}; mean within-task Spearman: {sanity['mean_within_dataset_spearman']:.4f}. Prefix diagnostics and collapse checks are retained in encoder_sanity.json.",'',
      '## Limits and conclusion','',
      'The final tasks are Iris, German Credit, and Sonar. Two validation tasks and this small test pool cannot support a generalization claim across domains. UCI source availability and dataset restrictions are documented in the audit, including pending access failures. No OpenML mirror-wide audit is claimed. Search times exclude bank generation and encoder training; offline transfer costs must be amortized in any deployment comparison. There is a per-trial timeout but no hard RAM cap.', '',
      'The software and this bounded experiment are reproducible. Positive or negative point estimates do not by themselves validate trajectory transfer. A larger independently collected benchmark, more tasks, sensitivity studies, and an externally fixed score target are needed for a confirmatory research claim.', '',
      '## Reproduction','',
      'Use the pinned environment, restore audited UCI data, run `python -m trajmeta.study --phase all`, then `python scripts/report_study.py`. Completed runs are resumed only when their signatures match; partial runs require inspection. Never alter the final protocol to fit test outcomes.'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,metric,title in zip(axes,['outer_macro_f1','n_target_capped'],['Outer macro-F1 (higher is better)','Capped target evaluations (lower is better)']):
        ax.barh(methods,[summary[m][metric] for m in methods])
        ax.set_title(title); ax.invert_yaxis(); ax.grid(axis='x',alpha=.25)
    fig.suptitle('Three held-out datasets, three seeds; descriptive means')
    fig.savefig(OUT/'results.png',dpi=160)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
