import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('report_study',Path(__file__).resolve().parents[1]/'scripts/report_study.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

def test_statistical_unit_is_dataset_not_seed():
    rows=[]
    for task,difference in [('a',1),('b',2),('c',3)]:
        for seed in [11,29,47]:
            rows.extend([dict(dataset_key=task,seed=seed,method='shaped',score=difference),dict(dataset_key=task,seed=seed,method='ppo',score=0)])
    result=report.paired(rows,'shaped','ppo','score')
    assert result['mean_difference']==2
    assert result['exact_two_sided_task_sign_flip_p']==.25
    assert len(result['task_differences'])==3
    rows[-1]['seed']=99
    with pytest.raises(AssertionError): report.paired(rows,'shaped','ppo','score')

def test_zero_effect_has_p_one():
    rows=[dict(dataset_key=t,seed=11,method=m,score=0) for t in ['a','b','c'] for m in ['shaped','ppo']]
    result=report.paired(rows,'shaped','ppo','score')
    assert result['exact_two_sided_task_sign_flip_p']==1
    assert result['task_bootstrap_95_interval']==[0,0]
