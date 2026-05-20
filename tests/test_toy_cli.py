from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOY = ROOT / "examples" / "toy"
MPIEXEC = shutil.which("mpiexec") or shutil.which("mpirun")


def mpi_available() -> bool:
    return MPIEXEC is not None and all(
        importlib.util.find_spec(name) is not None for name in ("mpi4py", "schwimmbad", "emcee")
    )


def subprocess_env():
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def copy_toy(tmp_path: Path):
    for name in ["toy_model.py", "posterior.py", "make_config.py"]:
        shutil.copy(TOY / name, tmp_path / name)


@pytest.mark.skipif(importlib.util.find_spec("emcee") is None, reason="emcee is not installed")
def test_toy_serial_timing_cli(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "black_box_bayes",
            "--input",
            "toy_config.pkl",
            "--posterior-module",
            "posterior",
            "--sampler",
            "emcee",
            "--chains",
            "8",
            "--steps",
            "20",
            "--no-mpi",
            "--serial-timing-test",
        ],
        cwd=tmp_path,
        env=subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "log_posterior" in result.stdout
    assert "hh:mm:ss" in result.stdout


@pytest.mark.skipif(importlib.util.find_spec("emcee") is None, reason="emcee is not installed")
def test_toy_emcee_cli_writes_idata(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    out = tmp_path / "toy_emcee_idata.nc"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black_box_bayes",
            "--input",
            "toy_config.pkl",
            "--posterior-module",
            "posterior",
            "--sampler",
            "emcee",
            "--chains",
            "12",
            "--steps",
            "80",
            "--burnin",
            "20",
            "--no-mpi",
            "--idata-results",
            str(out),
        ],
        cwd=tmp_path,
        env=subprocess_env(),
        check=True,
    )
    idata = az.from_netcdf(out)
    assert "theta" in idata.posterior
    assert idata.posterior["theta"].shape[-1] == 2
    mean = idata.posterior["theta"].mean(("chain", "draw")).values
    assert np.all(np.isfinite(mean))


@pytest.mark.skipif(not mpi_available(), reason="mpiexec, mpi4py, schwimmbad, and emcee are required")
def test_toy_mpi_timing_cli_distributes_work(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    result = subprocess.run(
        [
            MPIEXEC,
            "-n",
            "2",
            sys.executable,
            "-m",
            "black_box_bayes",
            "--input",
            "toy_config.pkl",
            "--posterior-module",
            "posterior",
            "--sampler",
            "emcee",
            "--chains",
            "4",
            "--steps",
            "10",
            "--require-mpi",
            "--MPI-timing-test",
        ],
        cwd=tmp_path,
        env=subprocess_env(),
        text=True,
        capture_output=True,
        check=True,
    )
    assert "emcee log_posterior samples on 1 workers" in result.stdout
    assert "hh:mm:ss" in result.stdout
