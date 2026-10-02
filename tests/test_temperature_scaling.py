"""Temperature scaling math: softens overconfidence, never invents signal."""

import numpy as np

from mesen.engine.temperature import brier_score, fit_temperature, softmax


def test_softmax_is_probabilities():
    rng = np.random.default_rng(0)
    probs = softmax(rng.normal(size=(20, 4)), 1.0)
    assert abs(probs.sum(axis=-1) - 1.0).max() < 1e-6


def test_overconfident_wrong_fits_hot_temperature():
    # Confident (logit gap 4) and wrong on every sample: the only fix is T > 1.
    logits = np.array([[4.0, 0.0, 0.0, 0.0]] * 10)
    labels = np.array([1] * 10)
    best_t, _ = fit_temperature(logits, labels)
    assert best_t > 1.0
    assert brier_score(logits, labels, best_t) < brier_score(logits, labels, 1.0)


def test_calibrated_model_keeps_temperature_near_one():
    # Labels sampled from the model's own distribution: the model is truly
    # calibrated, so the NLL optimum stays near T=1. (Deterministic argmax
    # labels would reward infinite sharpening instead.)
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(400, 4))
    probs = softmax(logits, 1.0)
    labels = np.array([int(rng.choice(4, p=row)) for row in probs])
    best_t, _ = fit_temperature(logits, labels)
    assert 0.5 < best_t < 2.0


def test_temperature_never_changes_argmax():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(30, 4))
    for t in (0.2, 1.0, 5.0):
        assert (softmax(logits, t).argmax(axis=-1) == logits.argmax(axis=-1)).all(), (
            "scaling must not flip decisions, only confidence"
        )
