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
    def log_prior(theta):
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
        (["--input", "cfg.pkl", "--ptemcee-ntemps", "0"], "--ptemcee-ntemps must be positive"),
        (["--input", "cfg.pkl", "--ptemcee-tmax", "1"], "--ptemcee-tmax must be greater than 1"),
        (
            ["--input", "cfg.pkl", "--dynesty-explore-batches", "-1"],
            "--dynesty-explore-batches must be non-negative",
        ),
        (
            [
                "--input", "cfg.pkl",
                "--sampler", "dynesty",
                "--dynesty-run", "static",
                "--dynesty-explore-batches", "3",
            ],
            "--dynesty-explore-batches requires --sampler dynesty --dynesty-run dynamic",
        ),
        (
            [
                "--input", "cfg.pkl",
                "--sampler", "dynesty",
                "--dynesty-run", "single",
                "--dynesty-explore-batches", "3",
            ],
            "--dynesty-explore-batches requires --sampler dynesty --dynesty-run dynamic",
        ),
        (
            [
                "--input", "cfg.pkl",
                "--sampler", "emcee",
                "--dynesty-explore-batches", "3",
            ],
            "--dynesty-explore-batches requires --sampler dynesty --dynesty-run dynamic",
        ),
        (
            ["--input", "cfg.pkl", "--dynesty-explore-nlive", "0"],
            "--dynesty-explore-nlive must be positive",
        ),
        (
            ["--input", "cfg.pkl", "--dynesty-explore-maxcall", "0"],
            "--dynesty-explore-maxcall must be positive",
        ),
        (["--input", "cfg.pkl", "--pocomc-n-total", "0"], "--pocomc-n-total must be positive"),
        (["--input", "cfg.pkl", "--pocomc-n-active", "0"], "--pocomc-n-active must be positive"),
        (["--input", "cfg.pkl", "--pocomc-n-evidence", "-1"], "--pocomc-n-evidence must be non-negative"),
        (
            ["--input", "cfg.pkl", "--pocomc-checkpoint-every", "-1"],
            "--pocomc-checkpoint-every must be non-negative",
        ),
        (
            ["--input", "cfg.pkl", "--pocomc-n-active", "512", "--pocomc-n-effective", "256"],
            "--pocomc-n-active must not exceed --pocomc-n-effective",
        ),
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


def test_warmup_validate_ptemcee_requires_log_prior(monkeypatch):
    monkeypatch.setattr(cli, "_ptemcee_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 2), dtype=float),
        log_likelihood=lambda theta: 0.0,
    )
    with pytest.raises(AttributeError, match="log_prior"):
        cli._warmup_and_validate(Namespace(sampler="ptemcee", chains=4, steps=5))


def test_warmup_validate_ptemcee_rejects_nan_log_prior(monkeypatch):
    monkeypatch.setattr(cli, "_ptemcee_imports", lambda: None)
    cli.posterior = SimpleNamespace(
        NDIM=2,
        starting_location=lambda n: np.zeros((n, 2), dtype=float),
        log_likelihood=lambda theta: 0.0,
        log_prior=lambda theta: np.nan,
    )
    with pytest.raises(ValueError, match="log_prior returned NaN"):
        cli._warmup_and_validate(Namespace(sampler="ptemcee", chains=4, steps=5))


def test_run_ptemcee_builds_ladder_and_writes_outputs(monkeypatch, tmp_path):
    cli.posterior = DummyPosterior
    records: dict[str, object] = {}

    class FakeChain:
        def __init__(self, p0):
            self.x = np.zeros((0, 3, 4, 2), dtype=float)
            self.logP = np.zeros((0, 3, 4), dtype=float)
            self.ensemble = SimpleNamespace(x=np.asarray(p0))

        def run(self, count):
            new = np.zeros((count, 3, 4, 2), dtype=float)
            self.x = np.concatenate([self.x, new], axis=0)
            self.logP = np.concatenate([self.logP, np.zeros((count, 3, 4))], axis=0)

        def get_acts(self):
            return np.ones((3, 2), dtype=float)

        def log_evidence_estimate(self):
            return -1.0, 0.1

    class FakeSampler:
        def __init__(self, nwalkers, ndim, logl, logp, betas=None, adaptive=False, scale_factor=2.0, mapper=map):
            records["nwalkers"] = nwalkers
            records["ndim"] = ndim
            records["betas"] = np.asarray(betas)
            records["adaptive"] = adaptive
            records["mapper"] = mapper

        def chain(self, p0, random=None, thin_by=None):
            records["p0_shape"] = np.asarray(p0).shape
            records["random"] = random
            return FakeChain(p0)

    def fake_make_ladder(ndim, ntemps=None, Tmax=None):
        records["ladder_ntemps"] = ntemps
        return np.array([1.0, 0.5, 0.1])

    monkeypatch.setattr(cli, "_ptemcee_imports", lambda: None)
    monkeypatch.setattr(cli, "ptemcee", SimpleNamespace(Sampler=FakeSampler, make_ladder=fake_make_ladder))
    monkeypatch.setattr(cli, "_ptemcee_to_inferencedata", lambda chain, args, runtime_seconds=None: "idata")
    monkeypatch.setattr(cli, "_write_idata", lambda idata, path: path)
    monkeypatch.setattr(cli, "_write_ptemcee_native_results", lambda chain, path: path)

    args = Namespace(
        input="fake.pkl",
        output=str(tmp_path),
        idata_results=None,
        chains=4,
        steps=10,
        burnin=0,
        batch_size=None,
        step_size=2.0,
        rtol=0.01,
        ptemcee_ntemps=3,
        ptemcee_tmax=None,
        ptemcee_adaptive=True,
        ptemcee_progress=False,
        ptemcee_native_results=None,
        no_ptemcee_native_results=False,
        seed=None,
    )

    cli.run_ptemcee(args, cli.SerialPool(), size=1)

    assert records["nwalkers"] == 4
    # No --seed: ptemcee is left to build its own RandomState.
    assert records["random"] is None
    assert records["ndim"] == 2
    assert records["ladder_ntemps"] == 3
    assert records["p0_shape"] == (3, 4, 2)  # (ntemps, nwalkers, ndim)
    assert records["adaptive"] is True


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


def _explore_args(tmp_path, **overrides):
    """Args for a dynamic dynesty run, with everything run_dynesty touches."""
    args = Namespace(
        input="cfg.pkl",
        output=str(tmp_path),
        sampler="dynesty",
        dynesty_run="dynamic",
        dynesty_checkpoint=None,
        dynesty_checkpoint_every=300.0,
        dynesty_resume=False,
        dynesty_progress=False,
        dynesty_equal_weight=True,
        dynesty_explore_batches=3,
        dynesty_explore_nlive=250,
        dynesty_explore_maxcall=1000,
        idata_results=None,
        dynesty_results=None,
        dynesty_native_results=None,
        no_dynesty_native_results=True,
        nlive=100,
        nlive_batch=None,
        maxiter=None,
        maxcall=None,
        dlogz=None,
        dlogz_init=0.01,
        maxbatch=None,
        n_effective=None,
        dynesty_pfrac=0.8,
        dynesty_use_stop=True,
        add_live=True,
        queue_size=1,
        seed=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class FakeDynestySampler:
    """Stand-in for dynesty's DynamicSampler, recording add_batch calls."""

    ncall = 4242

    def __init__(self):
        self.batch_calls: list[dict] = []
        self.results = {"logz": np.array([-3.0, -2.0, -1.5])}

    def run_nested(self, **kwargs):
        self.ran = kwargs

    def add_batch(
        self,
        nlive=500,
        dlogz=0.01,
        mode="weight",
        logl_bounds=None,
        maxcall=None,
        print_progress=True,
        checkpoint_file=None,
        checkpoint_every=None,
    ):
        self.batch_calls.append(
            dict(
                nlive=nlive,
                mode=mode,
                logl_bounds=logl_bounds,
                maxcall=maxcall,
                print_progress=print_progress,
                checkpoint_file=checkpoint_file,
                checkpoint_every=checkpoint_every,
            )
        )


def _patch_dynesty_run(monkeypatch, sampler):
    cli.posterior = DummyPosterior
    monkeypatch.setattr(cli, "_dynesty_imports", lambda: None)
    monkeypatch.setattr(cli, "_make_dynesty_sampler", lambda *a, **k: sampler)
    monkeypatch.setattr(cli, "_dynesty_to_inferencedata", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_write_idata", lambda idata, path: path)
    monkeypatch.setattr(cli, "_write_dynesty_native_results", lambda results, path: None)


def test_run_dynesty_appends_full_range_explore_batches(monkeypatch, tmp_path):
    sampler = FakeDynestySampler()
    _patch_dynesty_run(monkeypatch, sampler)

    cli.run_dynesty(_explore_args(tmp_path), pool=None, size=1)

    assert len(sampler.batch_calls) == 3
    for call in sampler.batch_calls:
        assert call["mode"] == "full"
        # dynesty raises RuntimeError if logl_bounds is combined with any mode
        # other than 'manual', so it must never be passed.
        assert call["logl_bounds"] is None
        assert call["nlive"] == 250
        assert call["maxcall"] == 1000
        assert call["checkpoint_file"] == str(tmp_path / "dynesty_checkpoint.pkl")
        assert call["checkpoint_every"] == 300.0


def test_run_dynesty_explore_nlive_falls_back_to_nlive_batch(monkeypatch, tmp_path):
    sampler = FakeDynestySampler()
    _patch_dynesty_run(monkeypatch, sampler)
    args = _explore_args(
        tmp_path, dynesty_explore_batches=1, dynesty_explore_nlive=None, nlive_batch=777
    )

    cli.run_dynesty(args, pool=None, size=1)

    assert sampler.batch_calls[0]["nlive"] == 777


def test_run_dynesty_explore_nlive_defaults_to_dynesty_default(monkeypatch, tmp_path):
    sampler = FakeDynestySampler()
    _patch_dynesty_run(monkeypatch, sampler)
    args = _explore_args(
        tmp_path, dynesty_explore_batches=1, dynesty_explore_nlive=None, nlive_batch=None
    )

    cli.run_dynesty(args, pool=None, size=1)

    # Omitted entirely rather than passed as None, so dynesty's own default applies.
    assert sampler.batch_calls[0]["nlive"] == 500


def test_run_dynesty_skips_explore_batches_by_default(monkeypatch, tmp_path):
    sampler = FakeDynestySampler()
    _patch_dynesty_run(monkeypatch, sampler)

    cli.run_dynesty(_explore_args(tmp_path, dynesty_explore_batches=0), pool=None, size=1)

    assert sampler.batch_calls == []


def test_run_dynesty_explore_batches_tolerate_missing_add_batch_kwargs(monkeypatch, tmp_path):
    """A dynesty version whose add_batch lacks checkpoint_every must still work."""

    class NarrowSampler(FakeDynestySampler):
        def add_batch(self, nlive=500, mode="weight", maxcall=None, print_progress=True):
            self.batch_calls.append(
                dict(nlive=nlive, mode=mode, maxcall=maxcall, print_progress=print_progress)
            )

    sampler = NarrowSampler()
    _patch_dynesty_run(monkeypatch, sampler)

    cli.run_dynesty(_explore_args(tmp_path, dynesty_explore_batches=2), pool=None, size=1)

    assert len(sampler.batch_calls) == 2
    assert all("checkpoint_every" not in call for call in sampler.batch_calls)


def test_run_dynesty_ignores_explore_batches_for_static_runs(monkeypatch, tmp_path):
    """parse_args rejects this combination, but run_dynesty must not rely on that."""
    sampler = FakeDynestySampler()
    _patch_dynesty_run(monkeypatch, sampler)

    cli.run_dynesty(_explore_args(tmp_path, dynesty_run="static"), pool=None, size=1)

    assert sampler.batch_calls == []


def test_dynesty_conversion_records_explore_batch_provenance():
    cli.posterior = DummyPosterior
    args = Namespace(
        input="fake.pkl",
        output=".",
        seed=123,
        dynesty_run="dynamic",
        dynesty_equal_weight=False,
        dynesty_explore_batches=4,
        dynesty_explore_nlive=None,
        dynesty_explore_maxcall=5000,
        nlive_batch=321,
    )
    results = SimpleNamespace(
        samples=np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        logwt=np.log(np.array([0.2, 0.3, 0.5])),
        logz=np.array([0.0]),
    )

    attrs = cli._dynesty_to_inferencedata(results, args, runtime_seconds=0.1).posterior.attrs

    assert attrs["dynesty_explore_batches"] == 4
    assert attrs["dynesty_explore_nlive"] == 321
    assert attrs["dynesty_explore_maxcall"] == 5000


def test_dynesty_conversion_explore_attrs_default_without_the_flags():
    """Older Namespaces that predate the flags must still convert."""
    cli.posterior = DummyPosterior
    args = Namespace(
        input="fake.pkl", output=".", seed=1, dynesty_run="static", dynesty_equal_weight=False
    )
    results = SimpleNamespace(
        samples=np.zeros((2, 2)), logwt=np.log([0.5, 0.5]), logz=np.array([0.0])
    )

    attrs = cli._dynesty_to_inferencedata(results, args).posterior.attrs

    assert attrs["dynesty_explore_batches"] == 0
    assert attrs["dynesty_explore_nlive"] == "None"


def test_parameter_names_reports_config_source():
    cli.posterior = DummyPosterior
    assert cli._parameter_names() == (["a", "b"], "config")


def test_parameter_names_generated_when_absent():
    cli.posterior = SimpleNamespace(NDIM=3)
    assert cli._parameter_names() == (["theta_0", "theta_1", "theta_2"], "generated")


def test_parameter_names_flags_length_mismatch(capsys):
    cli.posterior = SimpleNamespace(NDIM=2, PARAMETER_NAMES=["only_one"])

    names, source = cli._parameter_names()

    assert names == ["theta_0", "theta_1"]
    assert source == "generated_length_mismatch"
    assert "does not match posterior.NDIM" in capsys.readouterr().err


def test_posterior_from_config_warns_on_name_count_mismatch(tmp_path, capsys):
    config_path = tmp_path / "cfg.pkl"
    with config_path.open("wb") as handle:
        pickle.dump(SimpleNamespace(ndim=2, parameter_names=["only_one"]), handle)

    cli._posterior_from_config(config_path)

    # Warning must land before sampling starts, not after the run.
    assert "1 parameter names but NDIM is 2" in capsys.readouterr().err


# --- pocomc ----------------------------------------------------------------------------


def _box_posterior(**extra):
    attrs = dict(
        NDIM=2,
        log_likelihood=lambda theta: float(-0.5 * np.sum(np.asarray(theta) ** 2)),
        log_prior=lambda theta: 0.0 if np.all(np.abs(theta) <= 10) else -np.inf,
        prior_transform=lambda u: -10.0 + 20.0 * np.asarray(u, dtype=float),
    )
    attrs.update(extra)
    return SimpleNamespace(**attrs)


def test_pocomc_prior_infers_box_bounds():
    cli.posterior = _box_posterior()
    prior = cli._PocomcPrior(2, seed=0)
    assert np.array_equal(prior.bounds, [[-10.0, 10.0], [-10.0, 10.0]])
    assert prior.dim == 2
    x = prior.rvs(5)
    assert x.shape == (5, 2)
    assert np.all(prior.logpdf(x) == 0.0)


def test_pocomc_prior_unbounded_and_decreasing_axes():
    from statistics import NormalDist

    def transform(u):
        u = np.asarray(u, dtype=float)
        # Axis 0: Gaussian (infinite support). Axis 1: decreasing map onto [0, 2].
        x0 = -np.inf if u[0] <= 0 else np.inf if u[0] >= 1 else NormalDist().inv_cdf(u[0])
        return np.array([x0, 2.0 - 2.0 * u[1]])

    cli.posterior = _box_posterior(prior_transform=transform)
    bounds = cli._pocomc_prior_bounds(2)
    assert np.array_equal(bounds, [[-np.inf, np.inf], [0.0, 2.0]])


def test_pocomc_prior_explicit_bounds_override_inference():
    cli.posterior = _box_posterior(prior_bounds=[[-1.0, 1.0], [0.0, 5.0]])
    assert np.array_equal(cli._PocomcPrior(2).bounds, [[-1.0, 1.0], [0.0, 5.0]])
    cli.posterior = _box_posterior(prior_bounds=[[1.0, -1.0], [0.0, 5.0]])
    with pytest.raises(ValueError, match="lower < upper"):
        cli._PocomcPrior(2)


def test_pocomc_prior_sanity_flags_draws_outside_support():
    # A correlated transform: axis 1 depends on axis 0, so per-axis inference
    # underestimates its support.
    def transform(u):
        u = np.asarray(u, dtype=float)
        return np.array([u[0], u[1] + 3.0 * (u[0] - 0.5)])

    cli.posterior = _box_posterior(prior_transform=transform, log_prior=lambda theta: 0.0)
    with pytest.raises(ValueError, match="prior_bounds"):
        cli._pocomc_prior_sanity(cli._PocomcPrior(2, seed=1))


def test_pocomc_prior_sanity_flags_inconsistent_log_prior():
    cli.posterior = _box_posterior(log_prior=lambda theta: -np.inf)
    with pytest.raises(ValueError, match="same prior"):
        cli._pocomc_prior_sanity(cli._PocomcPrior(2, seed=1))


class FakePocomcSampler:
    """Stand-in for pocomc.Sampler with the pieces run_pocomc touches."""

    instances: list = []

    def __init__(self, prior, likelihood, n_dim=None, n_effective=512, n_active=256, pool=None,
                 flow="nsf6", precondition=True, sample="tpcn", n_steps=None, n_max_steps=None,
                 pytorch_threads=1, output_dir=None, output_label=None, random_state=None):
        self.prior = prior
        self.n_dim = n_dim
        self.n_effective = n_effective
        self.n_active = n_active
        self.pool = pool
        self.distribute = map if pool is None else pool.map
        self.init_kwargs = dict(flow=flow, output_dir=output_dir, output_label=output_label,
                                random_state=random_state)
        self.warmup = True
        self.t = 0
        self.calls = 0
        self.saved: list[str] = []
        FakePocomcSampler.instances.append(self)

    def save_state(self, path):
        self.saved.append(str(path))
        Path(path).write_text("state")

    restored_settings = None

    def load_state(self, path):
        self.__dict__.update(pool=None, distribute=map, t=7, calls=700, n_active=self.n_active,
                             progress=True, logz=-2.0, logz_err=0.2)
        if self.restored_settings is not None:
            self.bbb_settings = dict(self.restored_settings)

    def _compute_evidence(self, n=5000):
        self.evidence_calls = getattr(self, "evidence_calls", 0) + 1
        self.logz, self.logz_err = -1.0, 0.1
        return self.logz, self.logz_err

    def run(self, n_total=4096, n_evidence=4096, progress=True, resume_state_path=None, save_every=None):
        self.run_kwargs = dict(n_total=n_total, n_evidence=n_evidence, resume_state_path=resume_state_path,
                               save_every=save_every)
        if resume_state_path is not None:
            self.load_state(resume_state_path)
        self.t += 3
        self.calls += 300

    def evidence(self):
        return -1.25, 0.05

    def posterior(self, resample=False, trim_importance_weights=True, return_logw=False):
        x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        w = np.array([0.1, 0.2, 0.3, 0.4])
        logl = -np.arange(4.0)
        logp = np.zeros(4)
        if return_logw:
            return x, np.log(w), logl, logp
        return x, w, logl, logp

    @property
    def results(self):
        return {"x": np.zeros((4, 2)), "beta": np.ones(4), "blobs": None}


def _pocomc_args(tmp_path, **overrides):
    args = cli.parse_args(["--input", "fake.pkl", "--output", str(tmp_path), "--sampler", "pocomc"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def fake_pocomc(monkeypatch):
    FakePocomcSampler.instances = []
    cli.posterior = _box_posterior()
    monkeypatch.setattr(cli, "_pocomc_imports", lambda: None)
    monkeypatch.setattr(cli, "pocomc", SimpleNamespace(Sampler=FakePocomcSampler))
    return FakePocomcSampler


def test_pocomc_checkpoint_is_single_file_throttled_and_always_final(fake_pocomc, tmp_path):
    ckpt = tmp_path / "sub" / "ck.state"
    cls = cli._make_pocomc_sampler_class(ckpt, checkpoint_every=3600.0)
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2)

    sampler.warmup = False
    sampler.save_state(tmp_path / "pmc_5.state")  # within the interval: skipped
    assert sampler.saved == []
    sampler.save_state(tmp_path / "pmc_final.state")  # final: always written
    assert sampler.saved == [str(ckpt)]
    assert not (tmp_path / "pmc_final.state").exists()

    cls = cli._make_pocomc_sampler_class(ckpt, checkpoint_every=0.0)
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2)
    sampler.save_state(tmp_path / "pmc_1.state")  # warmup states would duplicate particles
    assert sampler.saved == []
    sampler.warmup = False
    sampler.save_state(tmp_path / "pmc_2.state")
    sampler.save_state(tmp_path / "pmc_3.state")
    assert sampler.saved == [str(ckpt), str(ckpt)]


def test_pocomc_load_state_keeps_current_pool(fake_pocomc, tmp_path, capsys):
    cls = cli._make_pocomc_sampler_class(tmp_path / "ck.state", checkpoint_every=0.0)
    pool = cli.SerialPool()
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2, pool=pool)
    sampler.load_state(tmp_path / "ck.state")
    assert sampler.pool is pool
    assert sampler.distribute == pool.map
    assert "iteration 7" in capsys.readouterr().out


def test_run_pocomc_fresh_run_writes_outputs(fake_pocomc, tmp_path):
    args = _pocomc_args(tmp_path, pocomc_n_total=1000, seed=3)
    cli.run_pocomc(args, cli.SerialPool(), size=1)

    sampler = fake_pocomc.instances[-1]
    assert sampler.pool is None  # serial: pocomc's own map
    assert sampler.run_kwargs["n_total"] == 1000
    assert sampler.run_kwargs["resume_state_path"] is None
    assert sampler.run_kwargs["save_every"] == 1
    assert sampler.init_kwargs["output_label"] == "pocomc_checkpoint"
    assert sampler.init_kwargs["random_state"] == 3
    assert (tmp_path / "pocomc_idata.nc").exists()
    native = np.load(tmp_path / "pocomc_results.npz")
    assert float(native["log_evidence"]) == -1.25
    assert "blobs" not in native.files


def test_run_pocomc_resumes_only_when_checkpoint_exists(fake_pocomc, tmp_path, capsys):
    args = _pocomc_args(tmp_path, pocomc_resume=True, no_pocomc_native_results=True)
    cli.run_pocomc(args, cli.SerialPool(), size=1)
    assert fake_pocomc.instances[-1].run_kwargs["resume_state_path"] is None
    assert "starting a fresh run" in capsys.readouterr().out

    ckpt = tmp_path / "pocomc_checkpoint.state"
    ckpt.write_text("state")
    cli.run_pocomc(args, cli.SerialPool(), size=1)
    sampler = fake_pocomc.instances[-1]
    assert sampler.run_kwargs["resume_state_path"] == str(ckpt)
    assert sampler.t == 10  # 7 restored + 3 new


def test_run_pocomc_warns_when_n_active_not_multiple_of_workers(fake_pocomc, tmp_path, capsys):
    args = _pocomc_args(tmp_path, pocomc_n_active=100, no_pocomc_native_results=True)
    pool = SimpleNamespace(map=map)
    cli.run_pocomc(args, pool, size=4)  # MPI-style: 3 workers
    assert fake_pocomc.instances[-1].pool is pool
    assert "not a multiple of the 3 workers" in capsys.readouterr().err


def test_pocomc_conversion_equal_weight_and_weighted(fake_pocomc, tmp_path):
    sampler = FakePocomcSampler(None, None, n_dim=2)
    sampler.t, sampler.calls = 12, 3456

    idata = cli._pocomc_to_inferencedata(sampler, _pocomc_args(tmp_path, seed=0), runtime_seconds=1.0)
    assert idata.posterior["theta"].shape == (1, 4, 2)
    assert np.allclose(idata.sample_stats["importance_weight"].values, 0.25)
    attrs = idata.posterior.attrs
    assert attrs["sampler"] == "pocomc"
    assert attrs["pocomc_log_evidence"] == -1.25
    assert attrs["pocomc_log_evidence_err"] == 0.05
    assert attrs["pocomc_iterations"] == 12
    assert attrs["pocomc_calls"] == 3456
    assert attrs["pocomc_final_ess"] == pytest.approx(1.0 / np.sum(np.array([0.1, 0.2, 0.3, 0.4]) ** 2))

    idata = cli._pocomc_to_inferencedata(sampler, _pocomc_args(tmp_path, pocomc_equal_weight=False))
    assert np.allclose(idata.sample_stats["importance_weight"].values[0], [0.1, 0.2, 0.3, 0.4])
    assert np.allclose(idata.sample_stats["log_likelihood"].values[0], -np.arange(4.0))


def test_dynesty_native_results_archive_format_unchanged(tmp_path):
    results = SimpleNamespace(samples=np.zeros((3, 2)), logz=np.array([-3.0, -2.0, -1.0]), bad=object())
    path = cli._write_dynesty_native_results(results, tmp_path / "nested" / "res.npz")
    data = np.load(path)
    assert str(data["_format"]) == "black_box_bayes dynesty native results npz"
    assert data["samples"].shape == (3, 2)


def test_pocomc_resume_warns_and_records_checkpointed_settings(fake_pocomc, tmp_path, capsys):
    cls = cli._make_pocomc_sampler_class(tmp_path / "ck.state", checkpoint_every=0.0, progress=False)
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2)
    args = _pocomc_args(tmp_path, pocomc_flow="maf3")
    sampler.bbb_settings = cli._pocomc_requested_settings(args)
    sampler.restored_settings = dict(sampler.bbb_settings, flow="nsf6")

    sampler.load_state(tmp_path / "ck.state")

    assert "checkpoint has flow=nsf6, but this run requested maf3" in capsys.readouterr().err
    assert sampler.progress is False  # this run's display flag, not the checkpoint's
    idata = cli._pocomc_to_inferencedata(sampler, args)
    assert idata.posterior.attrs["pocomc_flow"] == "nsf6"


def test_pocomc_resume_of_finished_state_reuses_evidence(fake_pocomc, tmp_path):
    cls = cli._make_pocomc_sampler_class(tmp_path / "ck.state", checkpoint_every=0.0)
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2)
    sampler.load_state(tmp_path / "ck.state")

    # No new iterations since the load: keep the checkpoint's estimate.
    assert sampler._compute_evidence(100) == (-2.0, 0.2)
    assert getattr(sampler, "evidence_calls", 0) == 0

    # The run went on sampling, so the evidence must be recomputed.
    sampler.t += 1
    assert sampler._compute_evidence(100) == (-1.0, 0.1)
    assert sampler.evidence_calls == 1


def test_pocomc_fresh_run_always_computes_evidence(fake_pocomc, tmp_path):
    cls = cli._make_pocomc_sampler_class(tmp_path / "ck.state", checkpoint_every=0.0)
    sampler = cls(cli._PocomcPrior(2), lambda x: 0.0, n_dim=2)
    sampler.logz, sampler.logz_err = -3.0, 0.3
    assert sampler._compute_evidence(100) == (-1.0, 0.1)


class _ResultsWithPersistentLogw(FakePocomcSampler):
    """pocomc's ``results``: per-iteration fields, but ``logw`` flat and toward beta = 1."""

    LOGL = np.array([[-5.0, -4.0, -3.0], [-2.0, -1.5, -1.0]])
    BETA = np.array([0.0, 1.0])
    LOGZ = np.array([0.0, -1.2])

    @property
    def results(self):
        logw = cli.pocomc_log_weights(self.LOGL, self.BETA, self.LOGZ).reshape(-1)
        return {"x": np.zeros((2, 3, 2)), "logl": self.LOGL, "beta": self.BETA, "logz": self.LOGZ,
                "logw": logw, "blobs": None}


def test_pocomc_native_archive_names_the_posterior_weights(fake_pocomc, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "pocomc", SimpleNamespace(Sampler=_ResultsWithPersistentLogw))
    cli.run_pocomc(_pocomc_args(tmp_path), cli.SerialPool(), size=1)
    native = np.load(tmp_path / "pocomc_results.npz")
    assert "logw" not in native.files  # the flat beta=1 array is not a per-iteration field
    assert native["logw_posterior"].shape == native["logl"].shape == (2, 3)
    assert float(native["logw_posterior_beta"]) == 1.0
    np.testing.assert_allclose(
        native["logw_posterior"], cli.pocomc_log_weights(native["logl"], native["beta"], native["logz"])
    )


def test_pocomc_log_weights_is_the_persistent_sampling_mixture():
    logl = np.array([[-5.0, -4.0], [-2.0, -1.0], [-1.5, -0.5]])
    beta = np.array([0.0, 0.4, 1.0])
    logz = np.array([0.0, -0.9, -1.7])
    for beta_final in (0.0, 0.25, 1.0):
        den = np.mean([np.exp(b * logl - z) for b, z in zip(beta, logz)], axis=0)
        want = np.exp(beta_final * logl) / den
        want /= want.sum()
        np.testing.assert_allclose(np.exp(cli.pocomc_log_weights(logl, beta, logz, beta_final)), want, rtol=1e-12)
    # flat input, iteration-major, gives the same weights flattened
    np.testing.assert_allclose(cli.pocomc_log_weights(logl.reshape(-1), beta, logz),
                               cli.pocomc_log_weights(logl, beta, logz).reshape(-1))


def test_pocomc_log_weights_matches_pocomc_itself():
    pocomc = pytest.importorskip("pocomc")
    from pocomc.particles import Particles

    rng = np.random.default_rng(0)
    particles = Particles(n_particles=4, n_dim=1)
    beta, logz = [0.0, 0.3, 1.0], [0.0, -0.7, -1.9]
    for b, z in zip(beta, logz):
        particles.update(dict(u=rng.normal(size=(4, 1)), x=rng.normal(size=(4, 1)), logdetj=np.zeros(4),
                              logl=rng.normal(-2, 1, size=4), logp=np.zeros(4), logw=np.zeros(4), blobs=None,
                              iter=0, logz=z, calls=4, steps=1, efficiency=1.0, ess=4.0, accept=0.5, beta=b))
    logl = particles.get("logl")
    for beta_final in (0.3, 0.6, 1.0):
        want, _ = particles.compute_logw_and_logz(beta_final)
        got = cli.pocomc_log_weights(logl, particles.get("beta"), particles.get("logz"), beta_final)
        np.testing.assert_allclose(got.reshape(-1), want, rtol=1e-12)
