"""Reference lightweight posterior interface expected by :mod:`black_box_bayes.cli`.

Projects normally provide their own top-level ``posterior.py`` module, then run
``black-box-bayes --posterior-module posterior --input config.pkl ...``. The module
must expose the functions shown below. This file is documentation-by-example and
is not used automatically by the CLI.
"""

from __future__ import annotations

try:
    import dill as pickle
except ImportError:  # pragma: no cover - dill is a declared dependency, fallback helps examples.
    import pickle
import numpy as np

_CONFIG = None
NDIM = None


def init_posterior(config_path):
    """Initialize the global CalibrationConfig-like object on this rank."""
    global _CONFIG, NDIM
    with open(config_path, "rb") as f:
        cfg = pickle.load(f)
    _CONFIG = cfg
    NDIM = cfg.ndim


def starting_location(nwalkers):
    """Generate starting locations for walkers/chains, shape ``(nwalkers, NDIM)``."""
    return _CONFIG.starting_location(nwalkers)


def log_posterior(theta):
    """Return log posterior density for one parameter vector, shape ``(NDIM,)``."""
    return _CONFIG.log_posterior(theta)


def log_likelihood(theta):
    """Return log likelihood for one parameter vector, shape ``(NDIM,)``."""
    return _CONFIG.log_likelihood(theta)


def prior_transform(u):
    """Transform one unit-cube vector, shape ``(NDIM,)``, into parameter space."""
    return _CONFIG.prior_transform(u)


def log_posterior_batch(thetas):
    """Return log posterior values for ``thetas`` with shape ``(nwalkers, NDIM)``."""
    return np.array([_CONFIG.log_posterior(theta) for theta in thetas])
