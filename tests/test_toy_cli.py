from __future__ import annotations

import functools
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


def _find_launcher() -> str | None:
    """Locate an MPI launcher belonging to the *running* interpreter's MPI stack.

    A launcher next to ``sys.executable`` is preferred: in a conda env or a venv built
    against a site MPI module, that is the one matching what ``mpi4py`` links against.
    Falling straight to ``PATH`` picks up whatever comes first, which on a cluster login
    node is typically a scheduler wrapper from an entirely different MPI stack --
    ``/usr/bin/mpiexec`` here is ``mpiexec.slurm``, and it launches N independent
    singletons that each believe they are rank 0.
    """
    bindir = Path(sys.executable).parent
    for name in ("mpiexec", "mpirun"):
        candidate = bindir / name
        if candidate.exists():
            return str(candidate)
    return shutil.which("mpiexec") or shutil.which("mpirun")


MPIEXEC = _find_launcher()


@functools.lru_cache(maxsize=1)
def mpi_available() -> bool:
    """True only if two ranks actually form one communicator.

    Checking that a launcher and the Python modules exist is not enough -- see
    ``_find_launcher``. The only reliable test is to launch and look at the size, so
    that a broken or mismatched stack skips these tests instead of failing them.
    """
    if MPIEXEC is None:
        return False
    if any(importlib.util.find_spec(n) is None for n in ("mpi4py", "schwimmbad", "emcee")):
        return False
    # Never burn an MPI launch inside someone's allocation just to probe.
    if os.environ.get("SLURM_JOB_ID"):
        return True
    try:
        probe = subprocess.run(
            [MPIEXEC, "-n", "2", sys.executable, "-c",
             "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_size())"],
            capture_output=True, text=True, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    sizes = [line.strip() for line in probe.stdout.splitlines() if line.strip().isdigit()]
    return probe.returncode == 0 and sizes == ["2", "2"]


def subprocess_env():
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def copy_toy(tmp_path: Path):
    for name in ["toy_model.py", "make_config.py"]:
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


@pytest.mark.skipif(importlib.util.find_spec("ptemcee") is None, reason="ptemcee is not installed")
def test_toy_ptemcee_cli_writes_idata_and_native_archive(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    out = tmp_path / "toy_ptemcee_idata.nc"
    native = tmp_path / "toy_ptemcee_results.npz"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "black_box_bayes",
            "--input",
            "toy_config.pkl",
            "--sampler",
            "ptemcee",
            "--chains",
            "8",
            "--steps",
            "200",
            "--ptemcee-ntemps",
            "4",
            "--no-mpi",
            "--idata-results",
            str(out),
            "--ptemcee-native-results",
            str(native),
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

    data = np.load(native)
    assert data["x"].ndim == 4  # (draw, ntemps, walker, dim)
    assert data["x"].shape[1] == 4  # ntemps
    assert "log_evidence" in data.files


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


# --- multiprocessing pool -------------------------------------------------------------
#
# The pool must be *faithful*: given the same work it must compute the same answer as
# serial. That is only checkable at a matched --queue-size, because dynesty's queue size
# changes how many points it pre-generates per iteration and therefore the order in which
# it consumes its RNG. Pin queue-size on both sides and the runs are bit-identical; leave
# it free and they differ by well under the quoted logz error, which is correct behaviour
# rather than a bug.


def _run_toy_dynesty(tmp_path, tag, *extra):
    out = tmp_path / tag
    subprocess.run(
        [
            sys.executable, "-m", "black_box_bayes",
            "--input", "toy_config.pkl",
            "--output", str(out),
            "--sampler", "dynesty",
            "--dynesty-run", "static",
            "--nlive", "100",
            "--dlogz", "1.0",
            "--seed", "7",
            "--dynesty-history", "none",
            "--dynesty-native-results", str(out / "res.npz"),
            "--idata-results", str(out / "idata.nc"),
            *extra,
        ],
        cwd=tmp_path, env=subprocess_env(), text=True, capture_output=True, check=True,
    )
    return np.load(out / "res.npz")


@pytest.mark.skipif(importlib.util.find_spec("dynesty") is None, reason="dynesty is not installed")
def test_multiprocessing_pool_matches_serial_at_equal_queue_size(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)

    serial = _run_toy_dynesty(tmp_path, "ser", "--pool", "serial", "--queue-size", "4")
    mp = _run_toy_dynesty(
        tmp_path, "mp", "--pool", "multiprocessing", "--nprocs", "4", "--queue-size", "4"
    )

    assert np.array_equal(serial["samples"], mp["samples"])
    assert float(serial["logz"][-1]) == float(mp["logz"][-1])
    assert int(np.sum(serial["ncall"])) == int(np.sum(mp["ncall"]))


@pytest.mark.skipif(importlib.util.find_spec("dynesty") is None, reason="dynesty is not installed")
def test_multiprocessing_pool_defaults_queue_size_to_nprocs(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    out = tmp_path / "auto"
    result = subprocess.run(
        [
            sys.executable, "-m", "black_box_bayes",
            "--input", "toy_config.pkl",
            "--output", str(out),
            "--sampler", "dynesty",
            "--pool", "multiprocessing", "--nprocs", "3",
            "--dynesty-run", "static", "--nlive", "100", "--dlogz", "1.0", "--seed", "7",
            "--dynesty-history", "none",
            "--idata-results", str(out / "idata.nc"),
        ],
        cwd=tmp_path, env=subprocess_env(), text=True, capture_output=True, check=True,
    )
    # All 3 workers evaluate -- there is no master rank sitting out, unlike MPI.
    assert "3 workers" in result.stdout
    assert "queue_size=3" in result.stdout
    assert (out / "idata.nc").exists()


@pytest.mark.skipif(importlib.util.find_spec("emcee") is None, reason="emcee is not installed")
def test_multiprocessing_pool_runs_emcee(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    out = tmp_path / "em"
    subprocess.run(
        [
            sys.executable, "-m", "black_box_bayes",
            "--input", "toy_config.pkl",
            "--output", str(out),
            "--sampler", "emcee",
            "--pool", "multiprocessing", "--nprocs", "2",
            "--chains", "8", "--steps", "50", "--burnin", "0",
            "--batch-size", "25", "--step-size", "2.0", "--rtol", "0.01",
            "--idata-results", str(out / "idata.nc"),
        ],
        cwd=tmp_path, env=subprocess_env(), text=True, capture_output=True, check=True,
    )
    posterior = az.from_netcdf(out / "idata.nc").posterior["theta"]
    assert posterior.sizes["chain"] == 8
    assert posterior.sizes["draw"] == 50
    assert np.isfinite(posterior.values).all()


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--no-mpi", "--pool", "mpi"], "--no-mpi contradicts"),
        (["--require-mpi", "--pool", "serial"], "--require-mpi contradicts"),
        (["--nprocs", "4", "--pool", "serial"], "--nprocs only applies"),
        (["--pool", "multiprocessing", "--nprocs", "0"], "--nprocs must be positive"),
    ],
)
def test_pool_argument_validation(tmp_path, flags, message):
    copy_toy(tmp_path)
    result = subprocess.run(
        [
            sys.executable, "-m", "black_box_bayes",
            "--input", "toy_config.pkl", "--sampler", "emcee", *flags,
        ],
        cwd=tmp_path, env=subprocess_env(), text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert message in result.stderr


# --- --seed reproducibility -----------------------------------------------------------
#
# Each backend draws its randomness from a different place, so "--seed makes the run
# reproducible" has to be checked per sampler rather than assumed from one of them:
# dynesty takes an rstate Generator, ptemcee a RandomState handed to chain(), and emcee
# an internal RandomState seeded from numpy's legacy *global* state at construction.
# emcee is the one that silently ignored --seed before, which is exactly why it is
# tested here rather than trusted.


def _toy_run(tmp_path, tag, sampler, *extra):
    out = tmp_path / tag
    subprocess.run(
        [
            sys.executable, "-m", "black_box_bayes",
            "--input", "toy_config.pkl",
            "--output", str(out),
            "--sampler", sampler,
            "--pool", "serial",
            "--idata-results", str(out / "idata.nc"),
            *extra,
        ],
        cwd=tmp_path, env=subprocess_env(), text=True, capture_output=True, check=True,
    )
    return az.from_netcdf(out / "idata.nc").posterior["theta"].values


EMCEE_ARGS = (
    "--chains", "16", "--steps", "100", "--burnin", "0",
    "--batch-size", "50", "--step-size", "2.0", "--rtol", "0.01",
)
PTEMCEE_ARGS = (
    "--chains", "8", "--ptemcee-ntemps", "6", "--ptemcee-tmax", "1e3",
    "--steps", "80", "--burnin", "10",
    "--batch-size", "40", "--step-size", "2.0", "--rtol", "0.01",
)


@pytest.mark.skipif(importlib.util.find_spec("emcee") is None, reason="emcee is not installed")
def test_emcee_seed_is_reproducible(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    a = _toy_run(tmp_path, "a", "emcee", *EMCEE_ARGS, "--seed", "11")
    b = _toy_run(tmp_path, "b", "emcee", *EMCEE_ARGS, "--seed", "11")
    c = _toy_run(tmp_path, "c", "emcee", *EMCEE_ARGS, "--seed", "99")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.skipif(importlib.util.find_spec("ptemcee") is None, reason="ptemcee is not installed")
def test_ptemcee_seed_is_reproducible(tmp_path):
    copy_toy(tmp_path)
    subprocess.run([sys.executable, "make_config.py"], cwd=tmp_path, check=True)
    a = _toy_run(tmp_path, "a", "ptemcee", *PTEMCEE_ARGS, "--seed", "11")
    b = _toy_run(tmp_path, "b", "ptemcee", *PTEMCEE_ARGS, "--seed", "11")
    c = _toy_run(tmp_path, "c", "ptemcee", *PTEMCEE_ARGS, "--seed", "99")
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_seed_supplies_pymc_random_seed(tmp_path):
    from black_box_bayes.cli import parse_args

    args = parse_args(["--input", "x.pkl", "--seed", "5"])
    assert args.pymc_random_seed == 5
    # An explicit --pymc-random-seed still wins.
    args = parse_args(["--input", "x.pkl", "--seed", "5", "--pymc-random-seed", "7"])
    assert args.pymc_random_seed == 7
    # No --seed leaves it unset rather than inventing one.
    args = parse_args(["--input", "x.pkl"])
    assert args.pymc_random_seed is None
