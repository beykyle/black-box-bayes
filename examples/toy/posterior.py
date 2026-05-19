"""Lightweight posterior interface consumed by black-box-bayes for the toy example."""

from __future__ import annotations

try:
    import dill as pickle
except ImportError:  # pragma: no cover - dill is a declared dependency, fallback helps examples.
    import pickle
import numpy as np

_CONFIG = None
NDIM = None
PARAMETER_NAMES = None


def init_posterior(config_path):
    global _CONFIG, NDIM, PARAMETER_NAMES
    with open(config_path, "rb") as f:
        _CONFIG = pickle.load(f)
    NDIM = _CONFIG.ndim
    PARAMETER_NAMES = getattr(_CONFIG, "parameter_names", [f"theta_{i}" for i in range(NDIM)])


def starting_location(nwalkers):
    return _CONFIG.starting_location(nwalkers)


def log_posterior(theta):
    return _CONFIG.log_posterior(theta)


def log_likelihood(theta):
    return _CONFIG.log_likelihood(theta)


def prior_transform(u):
    return _CONFIG.prior_transform(u)


def log_posterior_batch(thetas):
    return np.array([log_posterior(theta) for theta in thetas])
