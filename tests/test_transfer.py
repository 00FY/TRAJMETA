import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from trajmeta.agents import run_ppo
from trajmeta.automl import Evaluator, finite_f_classif
from trajmeta.cli import load_config
from trajmeta.data import load_dataset, split_task
from trajmeta.encoder import QualityLSTM, sequence, embed
from trajmeta.env import PipelineEnv
from trajmeta.study import final_test, run_unit

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture
def setup():
    config=load_config(ROOT/'configs/pilot.yaml')
    row=pd.read_csv(ROOT/'data/dataset_manifest.csv').iloc[0]
    task=split_task(*load_dataset(row,ROOT))
    return config,task

def ev(task):
    return Evaluator(task.X_train,task.y_train,budget=3,folds=2,seed=42)

class Cold:
    records=[]
    def query(self,*args,**kwargs):
        return dict(potential=0.,accepted=False,indices=[])
    initial=query

def test_forced_cold_preserves_ppo_exactly(setup):
    config,task=setup
    a,b=ev(task),ev(task)
    run_ppo(a,config['search_space'],42,config['ppo'])
    episodes=run_ppo(b,config['search_space'],42,config['ppo'],retriever=Cold(),strength=.3,warm_mode='learned')
    assert [(r['pipeline_config'],r['inner_score']) for r in a.records]==[(r['pipeline_config'],r['inner_score']) for r in b.records]
    assert all(s['shaping_difference']==0 for e in episodes for s in e['steps'])

def test_outcome_masking_and_prefix_batching():
    steps=[dict(state=[.1]*37,action=i,base_reward=0) for i in range(4)]
    changed=copy.deepcopy(steps)
    changed[-1]['base_reward']=1
    changed[-1]['next_state']=[1]*37
    assert np.array_equal(sequence(steps),sequence(changed))
    model=QualityLSTM()
    _,together=embed(model,[sequence(steps[:1]),sequence(steps)])
    _,alone=embed(model,[sequence(steps[:1])])
    np.testing.assert_allclose(together[:1],alone,atol=1e-7)
    with pytest.raises(ValueError): sequence(steps*2)
    changed[0]['state'][0]=float('nan')
    with pytest.raises(ValueError): sequence(changed)

def test_environment_shaping_telescopes(setup):
    config,task=setup
    class Fixed(Cold):
        def query(self,*args,**kwargs):
            return dict(potential=.7,accepted=True,indices=[])
    env=PipelineEnv(ev(task),config['search_space'],retriever=Fixed(),shaping_strength=.3,gamma=.99)
    env.reset()
    for _ in range(4): env.step(0)
    steps=env.episodes[0]['steps']
    assert steps[-1]['next_potential']==0
    assert sum(.99**i*s['shaping_difference'] for i,s in enumerate(steps))==pytest.approx(0,abs=1e-12)
    assert len(env.evaluator.records)==1

def test_constant_feature_scores_are_finite():
    X=np.array([[1,0,2],[1,1,4],[1,2,3],[1,3,9],[1,4,8],[1,5,7]])
    scores,p=finite_f_classif(X,[0,0,0,1,1,1])
    assert np.isfinite(scores).all() and np.isfinite(p).all()
    assert scores[0]==0

def test_final_protocol_checks_config_before_evaluation(tmp_path):
    with pytest.raises(ValueError,match='Configuration changed'):
        final_test(tmp_path,{'budget':32},None,{'config':{'budget':64}})

def test_unknown_phase_is_rejected(tmp_path):
    with pytest.raises(ValueError,match='Unknown study phase'):
        run_unit(tmp_path,{},None,'x',11,'ppo','unsealed')

def test_frozen_retrieval_rejects_mismatched_bank(setup,tmp_path):
    from hashlib import sha256
    from trajmeta.encoder import FEATURE_SCHEMA
    from trajmeta.retrieval import Retriever
    from trajmeta.trajectories import write_bank
    config,task=setup
    env=PipelineEnv(ev(task),config['search_space'])
    for action in [0,1,2]:
        env.reset()
        for _ in range(4): env.step(action)
    records=[dict(e,schema_version=1,trajectory_id=str(i),dataset_key='train',split='meta_train',seed=11,policy_version='test',provenance_hash='test') for i,e in enumerate(env.episodes)]
    bank=tmp_path/'bank.jsonl'; checkpoint=tmp_path/'encoder.pt'
    write_bank(bank,records)
    torch.save(dict(state_dict=QualityLSTM().state_dict(),schema=FEATURE_SCHEMA,hidden_size=32,bank_sha256=sha256(bank.read_bytes()).hexdigest()),checkpoint)
    retriever=Retriever(bank,checkpoint,threshold=-1)
    for n in range(1,5):
        result=retriever.query(records[0]['steps'][:n],env.meta)
        assert result['accepted'] and 0<=result['potential']<=1
        assert set(result['trajectory_ids'])<=set(r['trajectory_id'] for r in records)
    records[0]['trajectory_id']='modified'
    write_bank(bank,records)
    with pytest.raises(ValueError,match='versions differ'):
        Retriever(bank,checkpoint)
