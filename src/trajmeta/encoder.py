"""Shared prefix/full LSTM; terminal outcome is supervision, never an input."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr

from .trajectories import read_bank

FEATURE_SCHEMA = 'pre_state37+action_onehot12+masked_reward0+step_fraction_v1'


def sequence(steps, width=37, actions=12):
    if not 1 <= len(steps) <= 4:
        raise ValueError('Expected one to four trajectory tokens')
    result = []
    for i, step in enumerate(steps):
        state = np.asarray(step['state'], dtype=np.float32)
        if state.shape != (width,) or not np.isfinite(state).all() or not isinstance(step['action'], (int, np.integer)) or not -1 <= step['action'] < actions:
            raise ValueError('Encoder state/action schema mismatch')
        action = np.zeros(actions,dtype=np.float32) if step['action']==-1 else np.eye(actions, dtype=np.float32)[step['action']]
        result.append(np.r_[state, action, 0., (i+1)/4])
    if not result:
        raise ValueError('An embedding needs at least one prefix token')
    return np.asarray(result, dtype=np.float32)


def rank_targets(records):
    values = np.zeros(len(records), dtype=np.float32)
    for key in sorted({r['dataset_key'] for r in records}):
        ids = [i for i, r in enumerate(records) if r['dataset_key'] == key]
        scores = [records[i]['inner_score'] for i in ids]
        ranks = (rankdata(scores, method='average') - .5) / len(ids)
        values[ids] = ranks
    return values


class QualityLSTM(nn.Module):
    def __init__(self, input_size=51, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        z = hidden[-1]
        return self.head(z).squeeze(-1), z


def batch(sequences):
    lengths = torch.tensor([len(s) for s in sequences])
    array = np.zeros((len(sequences), int(lengths.max()), sequences[0].shape[1]), dtype=np.float32)
    for i, s in enumerate(sequences):
        array[i, :len(s)] = s
    return torch.from_numpy(array), lengths


def embed(model, sequences):
    model.eval()
    with torch.no_grad():
        scores, z = model(*batch(sequences))
    return scores.numpy(), z.numpy()


def simple_embedding(steps):
    """Nonlearned ordered action positions, plus training-data metadata."""
    result = np.zeros(7+4*12, dtype=np.float32)
    result[:7] = steps[0]['state'][:7]
    for i, step in enumerate(steps):
        if step['action'] >= 0:
            result[7+i*12+step['action']] = 1
    return result


def train(bank_path, output, epochs=100, seed=2026):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    records = read_bank(bank_path)
    if any(r['status'] != 'ok' for r in records):
        raise ValueError('Failed trials must be excluded explicitly before encoder supervision')
    train_records = [r for r in records if r['seed'] != 47]
    check_records = [r for r in records if r['seed'] == 47]
    if not train_records or not check_records:
        raise ValueError('Need training seeds and independent whole-run check seed 47')
    targets = rank_targets(train_records)
    seq, labels = [], []
    for r, y in zip(train_records, targets):
        seq.append(sequence([{'state':r['steps'][0]['state'],'action':-1}]))
        labels.append(y)
        for length in range(1, 5):
            seq.append(sequence(r['steps'][:length]))
            labels.append(y)
    x, lengths = batch(seq)
    y = torch.tensor(labels, dtype=torch.float32)
    model = QualityLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    logs = []
    for epoch in range(epochs):
        model.train()
        prediction, _ = model(x, lengths)
        loss = nn.functional.mse_loss(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        logs.append({'epoch': epoch+1, 'loss': float(loss.detach())})
    truth = []
    for r in check_records:
        reference = np.asarray([t['inner_score'] for t in train_records if t['dataset_key'] == r['dataset_key']])
        truth.append(float((np.sum(reference < r['inner_score']) + .5*np.sum(reference == r['inner_score'])) / len(reference)))
    scores, embeddings = embed(model, [sequence(r['steps']) for r in check_records])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarities = normalized @ normalized.T
    binary = np.asarray(truth) >= .5
    auc = float(roc_auc_score(binary, scores)) if len(set(binary)) == 2 else None
    correlations = {}
    for key in sorted({r['dataset_key'] for r in check_records}):
        ids = [i for i, r in enumerate(check_records) if r['dataset_key'] == key]
        corr = spearmanr(np.asarray(truth)[ids], scores[ids]).statistic
        correlations[key] = float(corr) if np.isfinite(corr) else None
    # Outcome-free four-stage action representation comparison via train-only kNN ranks.
    train_simple = np.asarray([simple_embedding(r['steps']) for r in train_records])
    check_simple = np.asarray([simple_embedding(r['steps']) for r in check_records])
    simple_scores = []
    for r, z in zip(check_records, check_simple):
        ids = np.asarray([i for i,t in enumerate(train_records) if t['dataset_key']==r['dataset_key']])
        nearest = ids[np.argsort(np.linalg.norm(train_simple[ids]-z, axis=1))[:5]]
        simple_scores.append(float(targets[nearest].mean()))
    simple_auc = float(roc_auc_score(binary, simple_scores)) if len(set(binary))==2 else None
    prefix_auc = {}
    for length in range(1, 5):
        prefix_scores, _ = embed(model, [sequence(r['steps'][:length]) for r in check_records])
        prefix_auc[str(length)] = float(roc_auc_score(binary, prefix_scores)) if len(set(binary)) == 2 else None
    mean_correlation = float(np.mean([v for v in correlations.values() if v is not None]))
    standard_deviation = float(embeddings.std(axis=0).mean())
    report = {'train_trajectories':len(train_records), 'check_trajectories':len(check_records),
              'training_seeds':sorted({r['seed'] for r in train_records}), 'check_seed':47,
              'feature_schema':FEATURE_SCHEMA, 'terminal_reward_masked':True,
              'prefix_quality_auc':prefix_auc, 'mean_within_dataset_spearman':mean_correlation,
              'quality_auc':auc, 'simple_knn_quality_auc':simple_auc,
              'within_dataset_spearman':correlations, 'embedding_std_mean':standard_deviation,
              'cosine_offdiagonal_mean':float(similarities[~np.eye(len(similarities),dtype=bool)].mean()),
              'check_mse':float(np.mean((scores-np.asarray(truth))**2)),
              'constant_rank_mse':float(np.mean((.5-np.asarray(truth))**2)),
              'gate_passed':bool(auc is not None and auc > .55 and mean_correlation > 0 and standard_deviation > .001 and logs[-1]['loss']<logs[0]['loss']),
              'limitation':'Whole-run seed holdout within training tasks; dataset-disjoint transfer checked separately on meta-validation'}
    torch.save({'state_dict':model.state_dict(),'schema':FEATURE_SCHEMA,'hidden_size':32,
                'bank_sha256':sha256(Path(bank_path).read_bytes()).hexdigest()}, output/'encoder.pt')
    (output/'training.json').write_text(json.dumps(logs,indent=2))
    (output/'sanity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


def load(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['schema'] != FEATURE_SCHEMA:
        raise ValueError('Encoder schema mismatch')
    model = QualityLSTM(hidden_size=checkpoint['hidden_size'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--bank',required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    train(a.bank,a.output)
