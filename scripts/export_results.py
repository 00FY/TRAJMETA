"""Export existing frozen-study results as flat CSVs without new inference."""
import csv
import json
from pathlib import Path

def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def main():
    out=Path(__file__).resolve().parents[1]/'reports/study-v1'
    stats=json.loads((out/'statistics.json').read_text())
    write(out/'method_summary.csv',[dict(method=m,**values) for m,values in stats['summary'].items()])
    flat=[]
    for task,methods in stats['per_dataset'].items():
        for method,metrics in methods.items():
            row=dict(dataset_key=task,method=method)
            row.update({f'{metric}_{name}':value for metric,values in metrics.items() for name,value in values.items()})
            flat.append(row)
    write(out/'per_dataset.csv',flat)
    runs=json.loads((out/'test_results.json').read_text())
    keys=['dataset_key','method','seed','evaluations','failures','best_inner_score','outer_macro_f1','outer_balanced_accuracy','search_seconds','target_score','n_target','n_target_capped']
    write(out/'test_runs.csv',[{**{k:r[k] for k in keys},'target_censored':r['n_target'] is None} for r in runs])
    print('Exported method_summary.csv, per_dataset.csv and test_runs.csv.')

if __name__=='__main__': main()
