"""Frozen training-only index, metadata shortlist, gated trajectory ranking."""
from hashlib import sha256
from pathlib import Path
import numpy as np
from .encoder import load, sequence, embed, simple_embedding, rank_targets
from .trajectories import read_bank
from .shaping import gated_potential


class Retriever:
    def __init__(self, bank_path, encoder_path, threshold=.8, top_k=5, temperature=.1,
                 mode='learned', force_cold=False):
        self.records = read_bank(bank_path)
        self.model, checkpoint = load(encoder_path)
        if sha256(Path(bank_path).read_bytes()).hexdigest()!=checkpoint['bank_sha256']:
            raise ValueError('Encoder and bank versions differ')
        self.threshold, self.top_k, self.temperature = threshold, top_k, temperature
        self.mode, self.force_cold = mode, force_cold
        ranks = rank_targets(self.records)
        self.positive = np.where(ranks >= .8)[0]
        self.metadata = {k:np.asarray(next(r for r in self.records if r['dataset_key']==k)['steps'][0]['state'][:7])
                         for k in sorted({r['dataset_key'] for r in self.records})}
        matrix = np.asarray(list(self.metadata.values()))
        self.mean, self.scale = matrix.mean(0), np.maximum(matrix.std(0), .01)
        _, self.embeddings = embed(self.model,[sequence(r['steps']) for r in self.records])
        self.simple = np.asarray([simple_embedding(r['steps']) for r in self.records])

    def shortlist(self, meta):
        ranked=sorted(self.metadata,key=lambda k:float(np.linalg.norm((self.metadata[k]-meta)/self.scale)))
        return set(ranked[:min(3,len(ranked))])

    def query(self, steps, meta, mode=None):
        if self.force_cold:
            return {'potential':0.,'accepted':False,'max_similarity':None,'trajectory_ids':[],'similarities':[],'indices':[]}
        selected=self.shortlist(np.asarray(meta))
        ids=np.asarray([i for i in self.positive if self.records[i]['dataset_key'] in selected],dtype=int)
        if not len(ids):
            return {'potential':0.,'accepted':False,'max_similarity':None,'trajectory_ids':[],'similarities':[],'indices':[]}
        mode=mode or self.mode
        if mode=='metadata':
            distances=np.asarray([np.linalg.norm((self.metadata[self.records[i]['dataset_key']]-meta)/self.scale) for i in ids])
            similarity=1/(1+distances)
        else:
            if mode=='simple':
                query=simple_embedding(steps)
                matrix=self.simple[ids]
            else:
                _, z=embed(self.model,[sequence(steps)])
                query=z[0]
                matrix=self.embeddings[ids]
            similarity=(matrix@query)/np.maximum(np.linalg.norm(matrix,axis=1)*np.linalg.norm(query),1e-8)
            similarity=np.clip(similarity,-1,1)
        order=np.argsort(-similarity,kind='stable')[:self.top_k]
        cutoff=-1 if mode=='metadata' else self.threshold
        accepted=order[similarity[order]>=cutoff]
        chosen=ids[accepted]
        return {'potential':gated_potential(similarity.tolist(),cutoff,self.temperature,self.top_k),
                'accepted':bool(len(chosen)), 'max_similarity':float(similarity.max()),
                'trajectory_ids':[self.records[i]['trajectory_id'] for i in chosen],
                'similarities':similarity[accepted].tolist(),'indices':chosen.tolist()}

    def initial(self, observation, mode=None):
        # A shared schema state-only query: no outcome, no action chosen yet.
        step={'state':np.asarray(observation).tolist(),'action':-1}
        return self.query([step],observation[:7],mode=mode)
