from __future__ import annotations

import importlib.util
import pickle
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import black_box_bayes.cli as cli


class DummyPosterior:
    NDIM = 2
    PARAMETER_NAMES = ["a", "b"]

    @staticmethod
    def starting_location(nwalkers):
        return np.zeros((nwalkers, 2), dtype=float)

    @staticmethod
    def log_posterior(theta):
        return float(-0.5 * np.sum(np.asarray(theta, dtype=float) ** 2))

    @staticmethod
    def log_likelihood(theta):
        return float(-0.5 * np.sum(np.asarray(theta, dtype=float) ** 2))

    @staticmethod
    def prior_transform(u):
        return np.asarray(u, dtype=float)


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--input", "cfg.pkl", "--no-mpi", "--require-mpi"], "--no-mpi and --require-mpi"),
        (["--input", "cfg.pkl", "--chains", "0"], "--chains must be positive"),
        (["--input", "cfg.pkl", "--steps", "0"], "--steps must be positive"),
        (["--input", "cfg.pkl", "--idata-thin", "0"], "--idata-thin must be positive"),
        (["--input", "cfg.pkl", "--queue-size", "0"], "--queue-size must be positive"),
        (["--input", "cfg.pkl", "--dynesty-pfrac", "1.5"], "--dynesty-pfrac must be between 0 and 1"),
    ],
)
def test_parse_args_rejects_invalid_inputs(argv, message, capsys):
    with pytest.raises(SystemExit):
        cli.parse_args(argv)
    captured = capsys.readouterr()
    assert message in captured.err


def test_idata_results_path_uses_deprecated_aliases(tmp_path):
    args = cli.parse_args(["--input", "cfg.pkl", "--dynesty-results", str(tmp_path / "dynesty_alias.nc")])
    assert cli._idata_results_path(args, "dynesty") == tmp_path / "dynesty_alias.nc"

    args = cli.parse_args(["--input", "cfg.pkl", "--pymc-results", str(tmp_path / "pymc_alias.nc")])
    assert cli._idata_results_path(args, "pymc") == tmp_path / "pymc_alias.nc"


def test_load_input_object_loads_local_module_from_input_directory(monkeypatch, tmp_path):
    module_name = "local_model"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "class LocalConfig:\n"
        "    ndim = 2\n"
        "    parameter_names = ['x', 'y']\n"
        "    def starting_location(self, nwalkers):\n"
        "        return [[0.0, 0.0] for _ in range(nwalkers)]\n"
        "    def log_posterior(self, theta):\n"
        "        return 0.0\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    cfg_path = tmp_path / "cfg.pkl"
    with cfg_path.open("wb") as f:
        pickle.dump(module.LocalConfig(), f)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [p for p in sys.path if Path(p or ".").resolve() != tmp_path.resolve()])
    sys.modules.pop(module_name, None)

    loaded = cli._load_input_object(cfg_path)

    assert loaded.__class__.__module__ == module_name
    assert loaded.parameter_names == ["x", "y"]


def test_posterior_from_config_normalizes_object_interface(monkeypatch, tmp_path):
    module_name = "local_model"
    module_path = tmp_path / f"{module_name}.py"
    module_path.write_text(
        "import numpy as np\n"
        "class LocalConfig:\n"
        "    ndim = 2\n"
        "    parameter_names = ['x', 'y']\n"
        "    def starting_location(self, nwalkers):\n"
        "        return np.zeros((nwalkers, 2), dtype=float)\n"
        "    def log_posterior(self, theta):\n"
        "        return float(-0.5 * np.sum(np.asarray(theta, dtype=float) ** 2))\n"
        "    def log_likelihood(self, theta):\n"
        "        return self.log_posterior(theta)\n"
        "    def prior_transform(self, u):\n"
        "        return np.asarray(u, dtype=float)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    cfg_path = tmp_path / "cfg.pkl"
    with cfg_path.open("wb") as f:
        pickle.dump(module.LocalConfig(), f)

    monkeypatch.chdir(tmp_path)
    posterior = cli._posterior_from_config(cfg_path)

    assert posterior.NDIM == 2
    assert posterior.PARAMETER_NAMES == ["x", "y"]
    np.testing.assert_allclose(posterior.starting_location(3), np.zeros((3, 2)))
    assert posterior.log_posterior(np.array([1.0, 2.0])) == pytest.approx(-2.5)


def test_warmup_validate_rejects_invalid_ndim():
    cli.posterior = SimpleNamespace(NDIM=0)
    with pytest.raises(ValueError, match="must be a positive integer"):
        cli._warmup_and_validate(Namespace(sampler="emcee", chains=4, steps=5))


def test_warmup_validate_rejects_bad_starting_location_shape(monkeypatch):
    monkeypatch.setattr(cli, "_emcee_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 3), dtype=float),
        log_posterior=lambda theta: 0.0,
    )
    with pytest.raises(ValueError, match="must return shape"):
        cli._warmup_and_validate(Namespace(sampler="emcee", chains=4, steps=5))


def test_warmup_validate_rejects_nonfinite_starting_location(monkeypatch):
    monkeypatch.setattr(cli, "_emcee_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.full((n, 2), np.nan),
        log_posterior=lambda theta: 0.0,
    )
    with pytest.raises(ValueError, match="non-finite"):
        cli._warmup_and_validate(Namespace(sampler="emcee", chains=4, steps=5))


def test_warmup_validate_rejects_nan_log_posterior(monkeypatch):
    monkeypatch.setattr(cli, "_emcee_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 2), dtype=float),
        log_posterior=lambda theta: np.nan,
    )
    with pytest.raises(ValueError, match="returned NaN"):
        cli._warmup_and_validate(Namespace(sampler="emcee", chains=4, steps=5))


def test_warmup_validate_dynesty_requires_hooks(monkeypatch):
    monkeypatch.setattr(cli, "_dynesty_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 2), dtype=float),
    )
    with pytest.raises(AttributeError, match="log_likelihood"):
        cli._warmup_and_validate(Namespace(sampler="dynesty", chains=None, steps=None))


def test_warmup_validate_dynesty_rejects_nan_log_likelihood(monkeypatch):
    monkeypatch.setattr(cli, "_dynesty_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 2), dtype=float),
        log_likelihood=lambda theta: np.nan,
        prior_transform=lambda u: np.asarray(u, dtype=float),
    )
    with pytest.raises(ValueError, match="log_likelihood returned NaN"):
        cli._warmup_and_validate(Namespace(sampler="dynesty", chains=None, steps=None))


def test_dynesty_conversion_preserves_weighted_sample_stats():
    cli.posterior = DummyPosterior
    args = Namespace(
        input="fake.pkl",
        output=".",
        seed=123,
        dynesty_run="static",
        dynesty_equal_weight=False,
    )
    results = SimpleNamespace(
        samples=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        logwt=np.log(np.array([0.2, 0.3, 0.5])),
        logz=np.array([0.0]),
        logl=np.array([-2.0, -1.0, -0.5]),
        niter=3,
        ncall=np.array([10, 20, 30]),
    )

    idata = cli._dynesty_to_inferencedata(results, args, runtime_seconds=0.1)

    assert idata.posterior["theta"].shape == (1, 3, 2)
    np.testing.assert_allclose(idata.sample_stats["importance_weight"].values[0], [0.2, 0.3, 0.5])
    np.testing.assert_allclose(idata.sample_stats["log_weight"].values[0], results.logwt)
    np.testing.assert_allclose(idata.sample_stats["log_likelihood"].values[0], results.logl)
    assert idata.posterior.attrs["dynesty_equal_weight_resample"] == 0
    assert idata.posterior.attrs["dynesty_posterior_draw_count"] == 3


def test_run_emcee_resumes_existing_backend(monkeypatch, tmp_path):
    cli.posterior = DummyPosterior
    backend_path = tmp_path / "chains.h5"
    backend_path.touch()
    records: dict[str, object] = {}

    class FakeBackend:
        iteration = 7

        def __init__(self, path):
            records["backend_path"] = Path(path)
            self.reset_called = False

        def reset(self, nwalkers, ndim):
            self.reset_called = True

        def get_last_sample(self):
            return SimpleNamespace(coords=np.full((4, 2), 1.5))

    class FakeSampler:
        iteration = 7

        def __init__(self, *args, **kwargs):
            records["sampler_backend"] = kwargs["backend"]

        def sample(self, p0, iterations, progress=False):
            records["resumed_p0"] = np.asarray(p0)
            self.iteration = iterations
            yield object()

        def get_autocorr_time(self, tol=0):
            raise RuntimeError("autocorr unavailable")

    monkeypatch.setattr(cli, "_emcee_imports", lambda: None)
    monkeypatch.setattr(cli, "_emcee_to_inferencedata", lambda backend, args, runtime_seconds=None: "idata")
    monkeypatch.setattr(cli, "_write_idata", lambda idata, path: path)
    monkeypatch.setattr(
        cli,
        "emcee",
        SimpleNamespace(
            backends=SimpleNamespace(HDFBackend=FakeBackend),
            EnsembleSampler=FakeSampler,
            moves=SimpleNamespace(StretchMove=lambda a: ("stretch", a)),
        ),
    )

    args = Namespace(
        input="fake.pkl",
        output=str(tmp_path),
        idata_results=None,
        emcee_backend="chains.h5",
        chains=4,
        steps=5,
        burnin=0,
        batch_size=None,
        step_size=2.0,
        rtol=0.01,
        emcee_progress=False,
    )

    cli.run_emcee(args, cli.SerialPool(), size=1)

    np.testing.assert_allclose(records["resumed_p0"], np.full((4, 2), 1.5))
    assert records["backend_path"] == backend_path
    assert records["sampler_backend"].reset_called is False


def test_run_emcee_resume_shape_mismatch_raises_runtime_error(monkeypatch, tmp_path):
    cli.posterior = DummyPosterior
    (tmp_path / "chains.h5").touch()

    class FakeBackend:
        iteration = 7

        def __init__(self, path):
            pass

        def get_last_sample(self):
            return SimpleNamespace(coords=np.zeros((3, 2), dtype=float))

    monkeypatch.setattr(cli, "_emcee_imports", lambda: None)
    monkeypatch.setattr(
        cli,
        "emcee",
        SimpleNamespace(
            backends=SimpleNamespace(HDFBackend=FakeBackend),
            EnsembleSampler=object,
            moves=SimpleNamespace(StretchMove=lambda a: ("stretch", a)),
        ),
    )

    args = Namespace(
        input="fake.pkl",
        output=str(tmp_path),
        idata_results=None,
        emcee_backend="chains.h5",
        chains=4,
        steps=5,
        burnin=0,
        batch_size=None,
        step_size=2.0,
        rtol=0.01,
        emcee_progress=False,
    )

    with pytest.raises(RuntimeError, match="Could not resume emcee backend"):
        cli.run_emcee(args, cli.SerialPool(), size=1)


def test_pool_context_falls_back_to_serial_when_mpi_unavailable(monkeypatch):
    monkeypatch.setattr(cli, "_mpi_imports", lambda required=False: False)
    pool, size, using_mpi = cli._pool_context(Namespace(no_mpi=False, require_mpi=False))
    assert isinstance(pool, cli.SerialPool)
    assert size == 1
    assert using_mpi is False


def test_pool_context_require_mpi_rejects_single_rank(monkeypatch):
    monkeypatch.setattr(cli, "_mpi_imports", lambda required=False: True)
    monkeypatch.setattr(
        cli,
        "MPI",
        SimpleNamespace(COMM_WORLD=SimpleNamespace(Get_size=lambda: 1)),
    )
    with pytest.raises(RuntimeError, match="only one MPI rank"):
        cli._pool_context(Namespace(no_mpi=False, require_mpi=True))


def test_pymc_initial_theta_uses_prior_mean():
    cli.posterior = SimpleNamespace(
        NDIM=2,
        prior_mean=lambda: np.array([1.0, -2.0], dtype=float),
    )
    theta = cli._pymc_initial_theta(Namespace(pymc_init="prior_mean", pymc_random_seed=None))
    np.testing.assert_allclose(theta, [1.0, -2.0])


def test_pymc_initial_theta_uses_starting_location():
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.array([[3.0, 4.0]], dtype=float),
    )
    theta = cli._pymc_initial_theta(Namespace(pymc_init="starting_location", pymc_random_seed=None))
    np.testing.assert_allclose(theta, [3.0, 4.0])


def test_pymc_initial_theta_random_prior_is_seeded_by_chain():
    cli.posterior = SimpleNamespace(
        NDIM=2,
        prior_transform=lambda u: np.asarray(u, dtype=float) * 10.0,
    )
    args = Namespace(pymc_init="random_prior", pymc_random_seed=123)

    theta0_first = cli._pymc_initial_theta(args, chain_index=0)
    theta0_second = cli._pymc_initial_theta(args, chain_index=0)
    theta1 = cli._pymc_initial_theta(args, chain_index=1)

    np.testing.assert_allclose(theta0_first, theta0_second)
    assert not np.allclose(theta0_first, theta1)
