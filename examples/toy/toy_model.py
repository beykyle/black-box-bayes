"""Tiny two-parameter Gaussian posterior used by the package tests/examples."""

from __future__ import annotations

try:
    import dill as pickle
except ImportError:  # pragma: no cover - dill is a declared dependency, fallback helps examples.
    import pickle
import numpy as np


class ToyGaussianConfig:
    """Gaussian likelihood with a broad Gaussian prior.

    Data are generated as ``y = theta + noise`` for a two-dimensional theta.
    The posterior is analytically Gaussian, which makes this useful for smoke
    tests without needing any domain-specific model code.
    """

    ndim = 2
    parameter_names = ["mu0", "mu1"]

    def __init__(self, y, sigma=0.2, prior_sigma=5.0, seed=123):
        self.y = np.asarray(y, dtype=float)
        self.sigma = float(sigma)
        self.prior_sigma = float(prior_sigma)
        self.seed = int(seed)

    def starting_location(self, nwalkers):
        rng = np.random.default_rng(self.seed)
        center = self.posterior_mean()
        return center + 1.0e-2 * rng.normal(size=(nwalkers, self.ndim))

    def log_likelihood(self, theta):
        theta = np.asarray(theta, dtype=float)
        resid = (self.y - theta) / self.sigma
        return -0.5 * np.sum(resid**2)

    def log_prior(self, theta):
        theta = np.asarray(theta, dtype=float)
        z = theta / self.prior_sigma
        return -0.5 * np.sum(z**2)

    def log_posterior(self, theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (self.ndim,):
            return -np.inf
        if not np.all(np.isfinite(theta)):
            return -np.inf
        return float(self.log_prior(theta) + self.log_likelihood(theta))

    def prior_transform(self, u):
        # Uniform unit cube to broad box prior used by dynesty. This transform
        # intentionally differs from the Gaussian prior above, so dynesty here
        # is primarily a driver smoke test rather than an exact same-prior demo.
        u = np.asarray(u, dtype=float)
        return -10.0 + 20.0 * u

    def posterior_mean(self):
        # Analytic posterior mean for Gaussian likelihood + zero-centered prior.
        inv_like = 1.0 / self.sigma**2
        inv_prior = 1.0 / self.prior_sigma**2
        return (self.y * inv_like) / (inv_like + inv_prior)


def make_config(path="toy_config.pkl"):
    cfg = ToyGaussianConfig(y=np.array([1.5, -0.6]), sigma=0.2, prior_sigma=5.0)
    with open(path, "wb") as f:
        pickle.dump(cfg, f)
    return cfg


if __name__ == "__main__":
    cfg = make_config()
    print("wrote toy_config.pkl")
    print("analytic posterior mean:", cfg.posterior_mean())
