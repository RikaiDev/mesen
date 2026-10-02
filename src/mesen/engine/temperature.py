"""Temperature scaling for decision-head logits.

A single temperature T applied as softmax(z / T). T > 1 softens
overconfident heads; T < 1 sharpens underconfident ones. Fit on NLL,
report Brier. Pure numpy, no model dependency.
"""

import numpy as np


def softmax(logits, temperature):
    shifted = (logits - logits.max(axis=-1, keepdims=True)) / temperature
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def nll_loss(logits, labels, temperature):
    probs = softmax(logits, temperature)
    return float(-np.log(probs[np.arange(len(labels)), labels] + 1e-12).mean())


def brier_score(logits, labels, temperature=1.0):
    probs = softmax(logits, temperature)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=-1).mean())


def fit_temperature(logits, labels, grid=None):
    """Grid-search T in log space; returns (best_T, best_NLL)."""
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(10.0), 60))
    best = min(((t, nll_loss(logits, labels, t)) for t in grid), key=lambda kv: kv[1])
    return float(best[0]), float(best[1])
