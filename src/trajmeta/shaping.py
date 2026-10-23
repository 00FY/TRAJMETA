"""Tested math only; learned retrieval is deliberately not wired into PPO yet."""
import numpy as np


def gated_potential(similarities, threshold, temperature=0.1, top_k=5):
    if not -1 <= threshold <= 1 or temperature <= 0 or top_k < 1:
        raise ValueError("Invalid gate settings")
    values = np.asarray(similarities, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(np.abs(values) > 1 + 1e-7):
        raise ValueError("Expected finite cosine similarities")
    accepted = np.sort(values[values >= threshold])[-top_k:]
    if not len(accepted):
        return 0.0
    weights = np.exp((accepted - accepted.max()) / temperature)
    weights /= weights.sum()
    return float(np.clip(np.dot(weights, (accepted + 1) / 2), 0, 1))


def shape_reward(base_reward, potential, next_potential, *, gamma=0.99, strength=0.0, terminal=False):
    if not np.isfinite([base_reward, potential, next_potential, gamma, strength]).all():
        raise ValueError("Non-finite reward component")
    if not 0 <= potential <= 1 or not 0 <= next_potential <= 1 or not 0 <= gamma <= 1 or strength < 0:
        raise ValueError("Invalid reward component range")
    effective_next = 0.0 if terminal else next_potential
    difference = strength * (gamma * effective_next - potential)
    return {"base_reward": base_reward, "potential": potential, "next_potential": effective_next,
            "shaping_difference": difference, "shaped_reward": base_reward + difference}
