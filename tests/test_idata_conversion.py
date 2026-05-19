from __future__ import annotations

from argparse import Namespace
import importlib.util
from types import SimpleNamespace

import arviz as az
import pytest
import numpy as np

import black_box_bayes.cli as cli


class DummyPosterior:
    NDIM = 2
    PARAMETER_NAMES = ["a", "b"]


class FakeEmceeBackend:
    iteration = 5

    def get_chain(self, discard=0, thin=1):
        # emcee convention: draw, walker, dim
        return np.arange(5 * 3 * 2, dtype=float).reshape(5, 3, 2)[discard::thin]

    def get_log_prob(self, discard=0, thin=1):
        return np.ones((5, 3), dtype=float)[discard::thin]


@pytest.mark.skipif(importlib.util.find_spec("h5netcdf") is None and importlib.util.find_spec("netCDF4") is None, reason="NetCDF4 writer backend is not installed")
def test_emcee_conversion_writes_arviz_netcdf(tmp_path):
    cli.posterior = DummyPosterior
    args = Namespace(
        input="fake.pkl",
        output=str(tmp_path),
        emcee_backend="chains.h5",
        idata_discard=1,
        idata_thin=2,
    )

    idata = cli._emcee_to_inferencedata(FakeEmceeBackend(), args, runtime_seconds=0.1)
    assert hasattr(idata, "to_netcdf")
    assert idata.posterior["theta"].shape == (3, 2, 2)  # walker, thinned draw, dim
    assert list(idata.posterior.coords["theta_dim"].values) == ["a", "b"]

    out = cli._write_idata(idata, tmp_path / "idata.nc")
    loaded = az.from_netcdf(out)
    assert loaded.posterior["theta"].shape == (3, 2, 2)


@pytest.mark.skipif(importlib.util.find_spec("h5netcdf") is None and importlib.util.find_spec("netCDF4") is None, reason="NetCDF4 writer backend is not installed")
def test_dynesty_conversion_writes_arviz_netcdf(tmp_path):
    cli.posterior = DummyPosterior
    args = Namespace(
        input="fake.pkl",
        output=str(tmp_path),
        seed=123,
        dynesty_run="static",
        dynesty_equal_weight=True,
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
    assert hasattr(idata, "to_netcdf")
    assert idata.posterior["theta"].shape == (1, 3, 2)
    out = cli._write_idata(idata, tmp_path / "dynesty.nc")
    loaded = az.from_netcdf(out)
    assert loaded.posterior["theta"].shape == (1, 3, 2)


def test_dynesty_native_results_archive_preserves_weighted_fields(tmp_path):
    results = SimpleNamespace(
        samples=np.array([[0.0, 0.0], [1.0, 1.0]]),
        logl=np.array([-2.0, -1.0]),
        logwt=np.log(np.array([0.4, 0.6])),
        logz=np.array([-0.2, 0.0]),
        logzerr=np.array([0.3, 0.1]),
        logvol=np.array([-0.5, -1.0]),
        ncall=np.array([5, 7]),
        custom_object=object(),
    )
    out = cli._write_dynesty_native_results(results, tmp_path / "dynesty_results.npz")
    assert out is not None
    data = np.load(out)
    assert set(["samples", "logl", "logwt", "logz", "logzerr", "logvol", "ncall"]).issubset(data.files)
    np.testing.assert_allclose(data["samples"], results.samples)
    assert "custom_object" not in data.files
    assert "_format" in data.files


def test_dynesty_native_results_path_default_and_disable(tmp_path):
    args = Namespace(output=str(tmp_path), dynesty_native_results=None, no_dynesty_native_results=False)
    assert cli._dynesty_native_results_path(args) == tmp_path / "dynesty_results.npz"
    args.dynesty_native_results = str(tmp_path / "custom.npz")
    assert cli._dynesty_native_results_path(args) == tmp_path / "custom.npz"
    args.no_dynesty_native_results = True
    assert cli._dynesty_native_results_path(args) is None
