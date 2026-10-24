"""Matched budgets with no cached/free repeat trials."""
import numpy as np

from .automl import candidates
from .env import PipelineEnv


def run_random(evaluator, space, seed):
    rng = np.random.default_rng(seed)
    configs = candidates(space)
    for _ in range(evaluator.budget):
        evaluator.evaluate(configs[int(rng.integers(len(configs)))])
    return []


def run_optuna(evaluator, space, seed):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    configs = candidates(space)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    def objective(trial):
        index = trial.suggest_categorical("configuration", list(range(len(configs))))
        return evaluator.evaluate(configs[index])["inner_score"]
    study.optimize(objective, n_trials=evaluator.budget)
    return []


def rollout_sizes(budget, settings):
    # SB3 rounds up to a full rollout; choose an exact divisor so it cannot overspend.
    total = budget * 4
    allowed = [n for n in range(4, min(total, settings["n_steps_max"]) + 1, 4) if total % n == 0]
    if not allowed or settings["batch_size_max"] < 2:
        raise ValueError("PPO needs n_steps_max >= 4 and batch_size_max >= 2")
    steps = max(allowed)
    batch = max(n for n in range(2, min(steps, settings["batch_size_max"]) + 1) if steps % n == 0)
    return total, steps, batch


def run_ppo(evaluator, space, seed, settings, model_path=None, retriever=None, strength=0., warm_mode=None):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    total, steps, batch = rollout_sizes(evaluator.budget, settings)
    env = PipelineEnv(evaluator, space, retriever=retriever, shaping_strength=strength,gamma=settings['gamma'])
    model = PPO("MlpPolicy", env, n_steps=steps, batch_size=batch, seed=seed, device="cpu",
                learning_rate=settings["learning_rate"], n_epochs=settings["n_epochs"],
                gamma=settings["gamma"], ent_coef=settings["ent_coef"],
                policy_kwargs={"net_arch": [32, 32]}, verbose=0)
    if retriever is not None and warm_mode:
        warm_start(model,env,retriever,warm_mode)
    model.learn(total_timesteps=total)
    if len(evaluator.records) != evaluator.budget:
        raise RuntimeError("PPO evaluation budget mismatch")
    if model_path is not None:
        model.save(model_path)
    env.close()
    return env.episodes


def warm_start(model,env,retriever,mode):
    """Offline behavior cloning on target-state templates; no target evaluations."""
    import torch
    from .automl import Candidate
    observation,_=env.reset()
    retrieval=retriever.initial(observation,mode='metadata' if mode in {'metadata','final_config'} else ('simple' if mode=='simple' else None))
    selected=[retriever.records[i] for i in retrieval['indices']]
    if mode=='final_config':
        # Retain just one final configuration from each shortlisted historical task.
        selected=[]
        for key in sorted(retriever.shortlist(env.meta)):
            group=[r for r in retriever.records if r['dataset_key']==key]
            selected.append(max(group,key=lambda r:r['inner_score']))
    if not selected:
        return
    states,actions=[],[]
    for record in selected:
        cfg=Candidate(**record['pipeline_config'])
        env.reset()
        for stage,choice in enumerate([cfg.scaler,cfg.selection,cfg.classifier,cfg.level]):
            states.append(env._observation())
            index=env.menus[stage].index(choice)
            actions.append(index)
            env.choices.append(index)
            env.previous=index
            env.stage+=1
    x=torch.as_tensor(np.asarray(states),dtype=torch.float32)
    y=torch.tensor(actions)
    optimizer=torch.optim.Adam(model.policy.parameters(),lr=.003)
    for _ in range(20):
        _, log_prob, _=model.policy.evaluate_actions(x,y)
        loss=-log_prob.mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.policy.parameters(),1.)
        optimizer.step()
    env.reset()
