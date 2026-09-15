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
