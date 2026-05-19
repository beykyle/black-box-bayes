#!/usr/bin/env python3
"""Unified black-box Bayesian inference driver.

This driver expects a lightweight ``posterior`` module exposing:

    init_posterior(config_path)
    starting_location(nwalkers)
    log_posterior(theta)
    log_likelihood(theta)
    prior_transform(u)       # required for dynesty
    log_posterior_batch(thetas)  # optional; used for timing only
    NDIM

All samplers write an ArviZ InferenceData NetCDF file.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib
import inspect
import os
import sys
from pathlib import Path
from time import time
from typing import Any, Iterable

import arviz as az
import numpy as np
import xarray as xr

# Posterior module is loaded at runtime from --posterior-module.
posterior = None

# Optional dependencies, imported only when needed.
emcee = None
dynesty = None
cloudpickle = None
pymc = None
pt = None
Op = None
MPIPool = None
MPI = None


class SerialPool:
    """Minimal pool with the API needed by emcee/dynesty/PyMC chain dispatch."""

    def map(self, func, iterable):
        return list(map(func, iterable))

    def is_master(self):
        return True

    def wait(self):
        return None

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def _import_optional(name: str, install_hint: str):
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise ImportError(f"{name} is required here. Install with `{install_hint}`.") from exc


def _emcee_imports():
    global emcee
    emcee = _import_optional("emcee", "pip install emcee")


def _dynesty_imports():
    global dynesty
    dynesty = _import_optional("dynesty", "pip install dynesty")


def _pymc_imports():
    global cloudpickle, pymc, pt, Op
    cloudpickle = _import_optional("cloudpickle", "pip install cloudpickle")
    pymc = _import_optional("pymc", "pip install pymc")
    pt = _import_optional("pytensor.tensor", "pip install pytensor")
    Op = importlib.import_module("pytensor.graph.op").Op


def _mpi_imports(required: bool = False):
    """Import MPI helpers if available; return True when MPI is usable."""
    global MPI, MPIPool
    try:
        MPI = importlib.import_module("mpi4py.MPI")
        MPIPool = importlib.import_module("schwimmbad").MPIPool
        return True
    except Exception as exc:
        # mpi4py can be installed even when no libmpi runtime is available;
        # that raises RuntimeError rather than ImportError. Treat it as
        # unavailable unless the user explicitly requires MPI.
        if required:
            raise ImportError(
                "MPI mode requires mpi4py, schwimmbad, and a loadable MPI runtime. "
                "Install/configure MPI and run under mpiexec."
            ) from exc
        return False


def _mpi_size() -> int:
    if _mpi_imports(required=False):
        return MPI.COMM_WORLD.Get_size()
    return 1


def _pool_context(args):
    """Return an MPI pool when requested/available, otherwise a serial pool."""
    if args.no_mpi:
        return SerialPool(), 1, False
    mpi_available = _mpi_imports(required=args.require_mpi)
    if not mpi_available:
        return SerialPool(), 1, False

    size = MPI.COMM_WORLD.Get_size()
    if size < 2:
        if args.require_mpi:
            raise RuntimeError(
                "--require-mpi was set, but only one MPI rank is available. "
                "Run with e.g. `mpiexec -n 4 python bayes_driver_unified.py ...`."
            )
        return SerialPool(), 1, False

    return MPIPool(), size, True


def _is_worker(pool, using_mpi: bool) -> bool:
    return using_mpi and not pool.is_master()


def _filter_kwargs_for_callable(func, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Drop kwargs not accepted by callable unless it has **kwargs."""
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return {k: v for k, v in kwargs.items() if v is not None}

    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return {k: v for k, v in kwargs.items() if v is not None}
    return {k: v for k, v in kwargs.items() if v is not None and k in params}


def _require_steps(args, sampler_name: str):
    if args.steps is None:
        raise ValueError(f"--steps is required for {sampler_name}.")


def _check_vector(name: str, value, ndim: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    expected = posterior.NDIM if ndim is None else ndim
    if arr.shape != (expected,):
        raise ValueError(f"{name} must have shape ({expected},), got {arr.shape}.")
    return arr


def _check_starting_location(n: int) -> np.ndarray:
    loc = np.asarray(posterior.starting_location(n), dtype=float)
    if loc.shape != (n, posterior.NDIM):
        raise ValueError(
            f"posterior.starting_location({n}) must return shape "
            f"({n}, {posterior.NDIM}), got {loc.shape}."
        )
    if not np.all(np.isfinite(loc)):
        raise ValueError("posterior.starting_location returned non-finite values.")
    return loc


def _check_logp(theta: np.ndarray):
    val = float(posterior.log_posterior(theta))
    if np.isnan(val):
        raise ValueError("posterior.log_posterior returned NaN at the warmup point.")
    return val


def _check_for_log_likelihood():
    if not hasattr(posterior, "log_likelihood"):
        raise AttributeError("posterior.py must expose log_likelihood(theta).")


def _check_for_prior_transform():
    if not hasattr(posterior, "prior_transform"):
        raise AttributeError("posterior.py must expose prior_transform(u) for dynesty.")


def _prior_mean_or_representative() -> np.ndarray:
    for attr in ("prior_mean", "PRIOR_MEAN"):
        if hasattr(posterior, attr):
            value = getattr(posterior, attr)
            return _check_vector(attr, value() if callable(value) else value)

    if hasattr(posterior, "prior_transform"):
        return _check_vector(
            "posterior.prior_transform(0.5)",
            posterior.prior_transform(np.full(posterior.NDIM, 0.5)),
        )

    return _check_starting_location(1)[0]


def _theta_coords_and_dims():
    names = None
    for attr in ("PARAMETER_NAMES", "parameter_names", "param_names"):
        if hasattr(posterior, attr):
            value = getattr(posterior, attr)
            names = value() if callable(value) else value
            break

    if names is None:
        names = [f"theta_{i}" for i in range(posterior.NDIM)]
    names = list(names)
    if len(names) != posterior.NDIM:
        print(
            "Parameter-name count does not match posterior.NDIM; using generic names.",
            file=sys.stderr,
        )
        names = [f"theta_{i}" for i in range(posterior.NDIM)]

    return {"theta_dim": names}, {"theta": ["theta_dim"]}


def _idata_results_path(args, sampler: str) -> Path:
    path = args.idata_results
    if path is None and sampler == "pymc":
        path = args.pymc_results
    if path is None and sampler == "dynesty":
        path = args.dynesty_results
    if path is None:
        path = Path(args.output) / f"{sampler}_idata.nc"
    return Path(path)


def _dynesty_native_results_path(args) -> Path | None:
    """Return the dynesty-native archive path, or None when disabled.

    This path is intentionally separate from ``--dynesty-results``, which is a
    deprecated alias for the standardized ArviZ InferenceData output.
    """
    if getattr(args, "no_dynesty_native_results", False):
        return None
    path = getattr(args, "dynesty_native_results", None)
    if path is None:
        path = Path(args.output) / "dynesty_results.npz"
    return Path(path)


def _write_dynesty_native_results(results, path: Path | None) -> Path | None:
    """Write dynesty's weighted/native result arrays to ``.npz``.

    ArviZ receives an equal-weight posterior view. This archive preserves the
    nested-sampling representation: weighted samples, log-evidence trajectory,
    likelihood-call accounting, live/dead point metadata, etc. We save all
    array-like scalar/string/numeric fields returned by ``results.asdict()``
    when available, falling back to the common public attributes.
    """
    if path is None:
        return None

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(results, "asdict"):
        try:
            raw = results.asdict()
        except Exception:
            raw = {}
    else:
        raw = {}

    if not raw:
        names = (
            "samples",
            "logl",
            "logwt",
            "logz",
            "logzerr",
            "logvol",
            "information",
            "niter",
            "ncall",
            "eff",
            "samples_id",
            "samples_it",
            "samples_u",
            "samples_bound",
            "batch_bounds",
            "batch_nlive",
            "scale",
        )
        raw = {name: getattr(results, name) for name in names if hasattr(results, name)}

    arrays: dict[str, np.ndarray] = {}
    skipped: list[str] = []
    for key, value in raw.items():
        if value is None:
            continue
        try:
            arr = np.asarray(value)
        except Exception:
            skipped.append(str(key))
            continue

        # Object arrays are fragile unless pickling is enabled. Keep the native
        # archive portable and non-pickled by storing only numeric, bool, and
        # string-like arrays.
        if arr.dtype.kind in "biufcUS":
            arrays[str(key)] = arr
        else:
            skipped.append(str(key))

    arrays["_format"] = np.asarray("black_box_bayes dynesty native results npz")
    arrays["_skipped_fields"] = np.asarray(skipped, dtype=str)
    np.savez_compressed(path, **arrays)
    return path


def _write_idata(idata, path: Path) -> Path:
    """Write either ArviZ 0.x InferenceData or ArviZ 1.x DataTree."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    idata.to_netcdf(path)
    return path



def _dataset_from_arrays(data: dict[str, np.ndarray] | None, coords: dict[str, Any] | None, dims: dict[str, list[str]] | None):
    """Build an xarray.Dataset with ArviZ-style chain/draw dimensions.

    Arrays are expected to have shape ``(chain, draw, ...)``. This helper is
    used for ArviZ 0.x, where ``InferenceData`` is xarray-Dataset based.
    """
    if not data:
        return None

    coords = dict(coords or {})
    dims = dict(dims or {})

    first = np.asarray(next(iter(data.values())))
    if first.ndim < 2:
        raise ValueError("InferenceData arrays must have at least chain and draw dimensions.")
    nchain, ndraw = first.shape[:2]
    coords.setdefault("chain", np.arange(nchain))
    coords.setdefault("draw", np.arange(ndraw))

    data_vars = {}
    for name, value in data.items():
        arr = np.asarray(value)
        if arr.ndim < 2:
            raise ValueError(f"Variable {name!r} must have at least chain/draw dimensions; got {arr.shape}.")
        var_dims = ["chain", "draw"] + list(dims.get(name, []))
        if len(var_dims) != arr.ndim:
            trailing = [f"{name}_dim_{i}" for i in range(arr.ndim - 2)]
            var_dims = ["chain", "draw"] + trailing
            for axis_name, axis_len in zip(trailing, arr.shape[2:]):
                coords.setdefault(axis_name, np.arange(axis_len))
        data_vars[name] = (tuple(var_dims), arr)

    return xr.Dataset(data_vars=data_vars, coords=coords)


def _inferencedata_from_arrays(
    *,
    posterior_data: dict[str, np.ndarray],
    sample_stats_data: dict[str, np.ndarray] | None = None,
    coords: dict[str, Any] | None = None,
    dims: dict[str, list[str]] | None = None,
    attrs: dict[str, Any] | None = None,
):
    """Create an ArviZ result across the 0.x -> 1.x API boundary.

    ArviZ 0.x exposes an ``InferenceData`` constructor that accepts xarray
    datasets. In ArviZ 1.x, ``arviz.InferenceData`` aliases xarray's
    ``DataTree`` instead, so we must build a tree of datasets directly.
    Both forms expose ``to_netcdf`` and can be loaded with ``az.from_netcdf``.
    """
    coords = coords or None
    dims = dims or None

    posterior_ds = _dataset_from_arrays(posterior_data, coords, dims)
    sample_stats_ds = _dataset_from_arrays(sample_stats_data, coords, {})
    groups = {"posterior": posterior_ds}
    if sample_stats_ds is not None:
        groups["sample_stats"] = sample_stats_ds

    inference_cls = az.__dict__.get("InferenceData")
    if inference_cls is not None and inference_cls is not xr.DataTree:
        idata = inference_cls(**groups)
    else:
        idata = xr.DataTree.from_dict(groups)

    if attrs:
        _add_attrs_to_idata(idata, attrs)
    return idata

def _netcdf_attr_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, (str, int, float, np.integer, np.floating)):
        return value.item() if hasattr(value, "item") else value
    if value is None:
        return "None"
    return str(value)


def _common_attrs(args, sampler: str, runtime_seconds=None, extra=None):
    attrs = {
        "sampler": sampler,
        "input": str(args.input),
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "posterior_ndim": int(posterior.NDIM),
        "driver": Path(__file__).name,
    }
    if runtime_seconds is not None:
        attrs["runtime_seconds"] = float(runtime_seconds)
    if extra:
        attrs.update(extra)
    return {k: _netcdf_attr_value(v) for k, v in attrs.items()}


def _add_attrs_to_idata(idata, attrs: dict[str, Any]):
    """Attach metadata to ArviZ 0.x InferenceData or ArviZ 1.x DataTree."""
    groups_obj = getattr(idata, "groups", None)

    if callable(groups_obj):
        # ArviZ 0.x InferenceData: groups() -> ["posterior", ...]
        for group_name in groups_obj():
            try:
                getattr(idata, group_name).attrs.update(attrs)
            except Exception:
                pass
    elif groups_obj is not None:
        # ArviZ 1.x DataTree: groups is a tuple like ("/", "/posterior", ...).
        for group_name in groups_obj:
            try:
                node = idata if group_name == "/" else idata[group_name]
                node.attrs.update(attrs)
            except Exception:
                pass
    else:
        try:
            idata.attrs.update(attrs)
        except Exception:
            pass

    return idata


def _emcee_to_inferencedata(backend, args, runtime_seconds=None) -> az.InferenceData:
    chain = np.asarray(backend.get_chain(discard=args.idata_discard, thin=args.idata_thin))
    if chain.ndim != 3:
        raise RuntimeError(f"Expected emcee chain shape (draw, walker, dim), got {chain.shape}.")

    # emcee stores (draw, walker, parameter). ArviZ expects (chain, draw, parameter).
    theta = np.moveaxis(chain, 0, 1)

    sample_stats = {}
    try:
        lp = np.asarray(backend.get_log_prob(discard=args.idata_discard, thin=args.idata_thin))
        if lp.shape == chain.shape[:2]:
            sample_stats["lp"] = np.moveaxis(lp, 0, 1)
    except Exception as exc:
        print(f"Could not include emcee log probabilities: {exc}", file=sys.stderr)

    coords, dims = _theta_coords_and_dims()
    attrs = _common_attrs(
        args,
        "emcee",
        runtime_seconds=runtime_seconds,
        extra={
            "emcee_backend": str(Path(args.output) / args.emcee_backend),
            "emcee_iteration": int(backend.iteration),
            "emcee_walkers_as_arviz_chains": True,
            "idata_discard": int(args.idata_discard),
            "idata_thin": int(args.idata_thin),
        },
    )
    return _inferencedata_from_arrays(
        posterior_data={"theta": theta},
        sample_stats_data=sample_stats or None,
        coords=coords,
        dims=dims,
        attrs=attrs,
    )


def _normalized_dynesty_weights(results) -> np.ndarray:
    logwt = np.asarray(results.logwt, dtype=float)
    logz = np.asarray(results.logz, dtype=float)
    if logwt.size == 0:
        raise RuntimeError("dynesty produced no weighted samples.")
    log_norm = logz[-1] if logz.size and np.isfinite(logz[-1]) else np.max(logwt)
    weights = np.exp(logwt - log_norm)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    total = weights.sum()
    if total <= 0:
        raise RuntimeError("dynesty posterior weights are all zero or non-finite.")
    return weights / total


def _dynesty_to_inferencedata(results, args, runtime_seconds=None) -> az.InferenceData:
    samples = np.asarray(results.samples, dtype=float)
    if samples.ndim != 2:
        raise RuntimeError(f"Expected dynesty samples shape (draw, dim), got {samples.shape}.")

    weights = _normalized_dynesty_weights(results)
    rng = np.random.default_rng(args.seed)

    if args.dynesty_equal_weight:
        try:
            utils = importlib.import_module("dynesty.utils")
            resampled = utils.resample_equal(samples, weights, rstate=rng)
        except Exception:
            idx = _systematic_resample_indices(weights, rng)
            resampled = samples[idx]
        theta = resampled[None, :, :]
        posterior_weights = np.full(theta.shape[1], 1.0 / theta.shape[1])
    else:
        theta = samples[None, :, :]
        posterior_weights = weights

    sample_stats = {"importance_weight": posterior_weights[None, :]}
    if hasattr(results, "logl"):
        raw_logl = np.asarray(results.logl, dtype=float)
        if raw_logl.shape[0] == theta.shape[1] and not args.dynesty_equal_weight:
            sample_stats["log_likelihood"] = raw_logl[None, :]
    if hasattr(results, "logwt") and not args.dynesty_equal_weight:
        sample_stats["log_weight"] = np.asarray(results.logwt, dtype=float)[None, :]

    extra = {
        "dynesty_run": str(args.dynesty_run),
        "dynesty_equal_weight_resample": bool(args.dynesty_equal_weight),
        "dynesty_raw_sample_count": int(samples.shape[0]),
        "dynesty_posterior_draw_count": int(theta.shape[1]),
    }
    for name in ("logz", "logzerr", "information", "niter", "ncall"):
        if hasattr(results, name):
            arr = np.asarray(getattr(results, name))
            if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                extra[f"dynesty_{name}"] = float(arr.reshape(-1)[0])
            elif arr.size > 1 and np.issubdtype(arr.dtype, np.number):
                extra[f"dynesty_{name}_final"] = float(arr.reshape(-1)[-1])

    coords, dims = _theta_coords_and_dims()
    return _inferencedata_from_arrays(
        posterior_data={"theta": theta},
        sample_stats_data=sample_stats,
        coords=coords,
        dims=dims,
        attrs=_common_attrs(args, "dynesty", runtime_seconds=runtime_seconds, extra=extra),
    )


def _systematic_resample_indices(weights: np.ndarray, rng) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _make_logposterior_op_class():
    if Op is None or pt is None:
        raise RuntimeError("PyMC/PyTensor imports have not been initialized.")

    class LogPosteriorOp(Op):
        """PyTensor Op wrapping posterior.log_posterior(theta)."""

        itypes = [pt.dvector]
        otypes = [pt.dscalar]

        def perform(self, node, inputs, outputs):
            theta = np.asarray(inputs[0], dtype=float)
            logp = float(posterior.log_posterior(theta))
            if np.isnan(logp):
                logp = -np.inf
            outputs[0][0] = np.asarray(logp, dtype=float)

    return LogPosteriorOp


def build_pymc_model(init_theta=None):
    LogPosteriorOp = _make_logposterior_op_class()
    logp_op = LogPosteriorOp()
    coords, _ = _theta_coords_and_dims()

    with pymc.Model(coords=coords) as model:
        # Flat variable; the actual prior is included in posterior.log_posterior.
        theta = pymc.Flat("theta", dims=("theta_dim",))
        pymc.Potential("black_box_log_posterior", logp_op(theta))
        if init_theta is not None:
            model.set_initval(theta, _check_vector("init_theta", init_theta))
    return model


def _pymc_initial_theta(args, chain_index=0):
    rng = np.random.default_rng(
        None if args.pymc_random_seed is None else args.pymc_random_seed + chain_index
    )
    if args.pymc_init == "prior_mean":
        return _prior_mean_or_representative()
    if args.pymc_init == "starting_location":
        return _check_starting_location(1)[0]
    if args.pymc_init == "random_prior":
        _check_for_prior_transform()
        return _check_vector("posterior.prior_transform(u)", posterior.prior_transform(rng.random(posterior.NDIM)))
    raise ValueError(f"Unknown PyMC init mode {args.pymc_init}.")


def _run_pymc_chain(payload):
    args_bytes, chain_index = payload
    args = cloudpickle.loads(args_bytes)
    _pymc_imports()

    draws = args.steps
    init_theta = _pymc_initial_theta(args, chain_index=chain_index)
    with build_pymc_model(init_theta=init_theta) as model:
        theta_var = model["theta"]
        step_name = args.pymc_step.lower()
        step_kwargs = {}
        if args.pymc_target_accept is not None:
            step_kwargs["target_accept"] = args.pymc_target_accept

        if step_name == "demetropolisz":
            step = pymc.DEMetropolisZ(vars=[theta_var], **_filter_kwargs_for_callable(pymc.DEMetropolisZ, step_kwargs))
        elif step_name == "demetropolis":
            step = pymc.DEMetropolis(vars=[theta_var])
        elif step_name == "metropolis":
            step = pymc.Metropolis(vars=[theta_var])
        else:
            raise ValueError(f"Unknown PyMC step method {args.pymc_step}.")

        seed = None if args.pymc_random_seed is None else args.pymc_random_seed + chain_index
        trace = pymc.sample(
            draws=draws,
            tune=args.pymc_tune,
            chains=1,
            cores=1,
            step=step,
            initvals={"theta": init_theta},
            random_seed=seed,
            progressbar=args.pymc_progress,
            compute_convergence_checks=False,
            return_inferencedata=True,
            discard_tuned_samples=True,
        )
    return trace


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to pickled CalibrationConfig-like object.")
    parser.add_argument(
        "--posterior-module",
        default="posterior",
        help=(
            "Import path for the lightweight posterior interface module. "
            "Defaults to a top-level module named 'posterior' in the current project."
        ),
    )
    parser.add_argument("--output", default="./", help="Output directory.")
    parser.add_argument("--idata-results", default=None, help="ArviZ InferenceData NetCDF output path.")
    parser.add_argument("--sampler", choices=["emcee", "dynesty", "pymc"], default="emcee")

    parser.add_argument("--no-mpi", action="store_true", help="Force serial execution even if MPI is installed.")
    parser.add_argument("--require-mpi", action="store_true", help="Fail unless running with mpi4py/schwimmbad and at least one worker rank.")

    parser.add_argument("--chains", type=int, default=None, help="emcee walkers or PyMC chains.")
    parser.add_argument("--pymc-chains", type=int, default=None, help="PyMC chains, overriding --chains.")
    parser.add_argument("--steps", type=int, default=None, help="MCMC draws/steps for emcee or PyMC.")

    parser.add_argument("--idata-discard", type=int, default=0, help="Discard this many emcee draws when exporting InferenceData.")
    parser.add_argument("--idata-thin", type=int, default=1, help="Thin emcee draws by this factor when exporting InferenceData.")

    parser.add_argument("--emcee-backend", default="chains.h5", help="emcee HDF5 backend filename inside --output.")
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--step-size", type=float, default=2.0)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--emcee-progress", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--serial-timing-test", action="store_true")
    parser.add_argument("--MPI-timing-test", action="store_true")

    parser.add_argument("--dynesty-run", choices=["static", "single", "dynamic"], default="static")
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--nlive-batch", type=int, default=None)
    parser.add_argument("--dynesty-bound", choices=["none", "single", "multi", "balls", "cubes"], default="multi")
    parser.add_argument("--dynesty-sample", choices=["auto", "unif", "rwalk", "slice", "rslice"], default="auto")
    parser.add_argument("--dynesty-walks", type=int, default=None)
    parser.add_argument("--dynesty-slices", type=int, default=None)
    parser.add_argument("--dynesty-facc", type=float, default=0.5)
    parser.add_argument("--dynesty-bootstrap", type=int, default=None)
    parser.add_argument("--dynesty-enlarge", type=float, default=None)
    parser.add_argument("--dynesty-update-interval", type=float, default=None)
    parser.add_argument("--dlogz", type=float, default=None)
    parser.add_argument("--dlogz-init", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--maxbatch", type=int, default=None)
    parser.add_argument("--n-effective", type=int, default=None)
    parser.add_argument("--dynesty-pfrac", type=float, default=0.8)
    parser.add_argument("--dynesty-use-stop", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--add-live", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dynesty-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--queue-size", type=int, default=None)
    parser.add_argument("--dynesty-checkpoint", type=str, default=None)
    parser.add_argument("--dynesty-checkpoint-every", type=float, default=300.0)
    parser.add_argument("--dynesty-resume", action="store_true")
    parser.add_argument("--dynesty-results", type=str, default=None, help="Deprecated alias for --idata-results in dynesty mode.")
    parser.add_argument(
        "--dynesty-native-results",
        type=str,
        default=None,
        help=(
            "Dynesty-native weighted results archive (.npz). Defaults to "
            "output/dynesty_results.npz for dynesty runs. This is separate from "
            "the standardized ArviZ InferenceData output."
        ),
    )
    parser.add_argument(
        "--no-dynesty-native-results",
        action="store_true",
        help="Disable the extra dynesty-native .npz archive; ArviZ output is still written.",
    )
    parser.add_argument("--dynesty-history", type=str, default=None, help="Dynesty evaluation-history file; use 'none' to disable.")
    parser.add_argument("--dynesty-equal-weight", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--pymc-tune", type=int, default=1000)
    parser.add_argument("--pymc-step", choices=["demetropolisz", "demetropolis", "metropolis"], default="demetropolisz")
    parser.add_argument("--pymc-init", choices=["prior_mean", "starting_location", "random_prior"], default="prior_mean")
    parser.add_argument("--pymc-progress", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--pymc-target-accept", type=float, default=None)
    parser.add_argument("--pymc-results", type=str, default=None, help="Deprecated alias for --idata-results in PyMC mode.")
    parser.add_argument("--pymc-random-seed", type=int, default=None)

    args = parser.parse_args(argv)

    positive = [
        ("--chains", args.chains),
        ("--pymc-chains", args.pymc_chains),
        ("--steps", args.steps),
        ("--batch-size", args.batch_size),
        ("--idata-thin", args.idata_thin),
        ("--nlive", args.nlive),
        ("--nlive-batch", args.nlive_batch),
        ("--queue-size", args.queue_size),
    ]
    for name, value in positive:
        if value is not None and value <= 0:
            parser.error(f"{name} must be positive.")
    if args.burnin < 0:
        parser.error("--burnin must be non-negative.")
    if args.idata_discard < 0:
        parser.error("--idata-discard must be non-negative.")
    if args.pymc_tune < 0:
        parser.error("--pymc-tune must be non-negative.")
    if not 0.0 <= args.dynesty_pfrac <= 1.0:
        parser.error("--dynesty-pfrac must be between 0 and 1.")
    if args.no_mpi and args.require_mpi:
        parser.error("--no-mpi and --require-mpi are mutually exclusive.")

    return args


def _require_emcee_args(args):
    if args.chains is None:
        raise ValueError("--chains is required for emcee; it is the number of walkers.")
    _require_steps(args, "emcee")
    if args.chains < 2 * posterior.NDIM:
        raise ValueError(
            f"emcee StretchMove needs at least 2*ndim walkers; got {args.chains}, ndim={posterior.NDIM}."
        )


def run_emcee(args, pool, size=1):
    _emcee_imports()
    _require_emcee_args(args)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    backend_path = output_path / args.emcee_backend
    idata_path = _idata_results_path(args, "emcee")

    backend = emcee.backends.HDFBackend(backend_path)
    fresh_backend = not backend_path.exists() or backend.iteration == 0

    if fresh_backend:
        p0 = _check_starting_location(args.chains)
        if args.burnin:
            print(f"Running emcee burn-in for {args.burnin} steps without writing backend...")
            burn_sampler = emcee.EnsembleSampler(
                args.chains,
                posterior.NDIM,
                posterior.log_posterior,
                pool=pool,
                moves=emcee.moves.StretchMove(a=args.step_size),
            )
            state = burn_sampler.run_mcmc(p0, args.burnin, progress=args.emcee_progress)
            p0 = state.coords
            print("Burn-in complete; initializing production backend.")
        backend.reset(args.chains, posterior.NDIM)
    else:
        try:
            last = backend.get_last_sample()
            p0 = last.coords
            if p0.shape != (args.chains, posterior.NDIM):
                raise ValueError(
                    f"Existing backend has walker/dim shape {p0.shape}, expected {(args.chains, posterior.NDIM)}."
                )
            print(f"Resuming emcee backend from iteration {backend.iteration}.")
        except Exception as exc:
            raise RuntimeError(
                f"Could not resume emcee backend {backend_path}. Delete it or change --emcee-backend."
            ) from exc

    sampler = emcee.EnsembleSampler(
        args.chains,
        posterior.NDIM,
        posterior.log_posterior,
        pool=pool,
        backend=backend,
        moves=emcee.moves.StretchMove(a=args.step_size),
    )

    chunk = args.batch_size or args.steps
    old_tau = None
    t0 = time()
    print(f"Running emcee: walkers={args.chains}, steps={args.steps}, ndim={posterior.NDIM}, mpi_ranks={size}.")
    sys.stdout.flush()

    for _ in sampler.sample(p0, iterations=args.steps, progress=args.emcee_progress):
        if sampler.iteration % chunk:
            continue
        try:
            tau = sampler.get_autocorr_time(tol=0)
        except Exception as exc:
            print(f"Autocorr estimate unavailable at step {sampler.iteration}: {exc}")
            sys.stdout.flush()
            continue
        print(f"Step {sampler.iteration}: mean autocorr time = {np.mean(tau):.2f}")
        if old_tau is not None:
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / np.maximum(tau, 1e-300) < args.rtol)
            if converged:
                print(f"Chains converged after {sampler.iteration} backend iterations.")
                break
        old_tau = tau
        sys.stdout.flush()

    dt = time() - t0
    idata = _emcee_to_inferencedata(backend, args, runtime_seconds=dt)
    _write_idata(idata, idata_path)
    print(f"emcee sampling took {_dt.timedelta(seconds=dt)}")
    print(f"Saved ArviZ InferenceData to {idata_path}")


def _make_dynesty_sampler(args, pool, queue_size: int, output_path: Path):
    _dynesty_imports()
    _check_for_log_likelihood()
    _check_for_prior_transform()
    rstate = np.random.default_rng(args.seed) if args.seed is not None else None

    common_kwargs = dict(
        bound=args.dynesty_bound,
        sample=args.dynesty_sample,
        pool=pool,
        queue_size=queue_size,
        rstate=rstate,
        walks=args.dynesty_walks,
        slices=args.dynesty_slices,
        facc=args.dynesty_facc,
        bootstrap=args.dynesty_bootstrap,
        enlarge=args.dynesty_enlarge,
        update_interval=args.dynesty_update_interval,
    )

    history = args.dynesty_history
    if history is None:
        history = output_path / "dynesty_history.h5"
    elif str(history).lower() == "none":
        history = None
    else:
        history = Path(history)
    if history is not None:
        common_kwargs.update(save_evaluation_history=True, history_filename=str(history))

    dynesty_run = "static" if args.dynesty_run == "single" else args.dynesty_run
    if dynesty_run == "dynamic":
        cls = dynesty.DynamicNestedSampler
        kwargs = _filter_kwargs_for_callable(cls, common_kwargs)
        return cls(posterior.log_likelihood, posterior.prior_transform, posterior.NDIM, **kwargs)

    if dynesty_run == "static":
        cls = dynesty.NestedSampler
        kwargs = dict(common_kwargs, nlive=args.nlive)
        kwargs = _filter_kwargs_for_callable(cls, kwargs)
        return cls(posterior.log_likelihood, posterior.prior_transform, posterior.NDIM, **kwargs)

    raise ValueError(f"Unknown dynesty run type {args.dynesty_run}.")


def _restore_dynesty_sampler(args, checkpoint_path: Path, pool):
    _dynesty_imports()
    dynesty_run = "static" if args.dynesty_run == "single" else args.dynesty_run
    if dynesty_run == "dynamic":
        return dynesty.DynamicNestedSampler.restore(str(checkpoint_path), pool=pool)
    if dynesty_run == "static":
        return dynesty.NestedSampler.restore(str(checkpoint_path), pool=pool)
    raise ValueError(f"Unknown dynesty run type {args.dynesty_run}.")


def run_dynesty(args, pool, size=1):
    _dynesty_imports()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.dynesty_checkpoint) if args.dynesty_checkpoint else output_path / "dynesty_checkpoint.pkl"
    idata_path = _idata_results_path(args, "dynesty")
    native_path = _dynesty_native_results_path(args)
    queue_size = args.queue_size or max(size - 1, 1)

    print(
        f"Running dynesty: run={args.dynesty_run}, nlive={args.nlive}, ndim={posterior.NDIM}, "
        f"queue_size={queue_size}, mpi_ranks={size}."
    )
    print(f"Dynesty native results: {native_path if native_path is not None else 'disabled'}")
    if args.dynesty_resume and checkpoint_path.exists():
        print(f"Restoring dynesty checkpoint from {checkpoint_path}")
        sampler = _restore_dynesty_sampler(args, checkpoint_path, pool)
        resume = True
    else:
        sampler = _make_dynesty_sampler(args, pool, queue_size, output_path)
        resume = False

    t0 = time()
    dynesty_run = "static" if args.dynesty_run == "single" else args.dynesty_run
    if dynesty_run == "static":
        run_kwargs = dict(
            maxiter=args.maxiter,
            maxcall=args.maxcall,
            dlogz=args.dlogz,
            add_live=args.add_live,
            print_progress=args.dynesty_progress,
            checkpoint_file=str(checkpoint_path),
            checkpoint_every=args.dynesty_checkpoint_every,
            resume=resume,
        )
    else:
        run_kwargs = dict(
            nlive_init=args.nlive,
            nlive_batch=args.nlive_batch,
            maxiter_init=args.maxiter,
            maxcall_init=args.maxcall,
            dlogz_init=args.dlogz_init,
            maxiter=args.maxiter,
            maxcall=args.maxcall,
            maxbatch=args.maxbatch,
            n_effective=args.n_effective,
            wt_kwargs={"pfrac": args.dynesty_pfrac},
            stop_kwargs={"pfrac": args.dynesty_pfrac},
            use_stop=args.dynesty_use_stop,
            print_progress=args.dynesty_progress,
            checkpoint_file=str(checkpoint_path),
            checkpoint_every=args.dynesty_checkpoint_every,
            resume=resume,
        )
    run_kwargs = _filter_kwargs_for_callable(sampler.run_nested, run_kwargs)
    sampler.run_nested(**run_kwargs)

    dt = time() - t0
    idata = _dynesty_to_inferencedata(sampler.results, args, runtime_seconds=dt)
    _write_idata(idata, idata_path)
    written_native_path = _write_dynesty_native_results(sampler.results, native_path)
    print(f"dynesty sampling took {_dt.timedelta(seconds=dt)}")
    print(f"Saved ArviZ InferenceData to {idata_path}")
    if written_native_path is not None:
        print(f"Saved dynesty-native weighted results to {written_native_path}")
    try:
        print(sampler.results.summary())
    except Exception:
        pass


def run_pymc(args, pool, size=1):
    _pymc_imports()
    _require_steps(args, "pymc")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    idata_path = _idata_results_path(args, "pymc")
    nchains = args.pymc_chains or args.chains or max(size - 1, 1)

    print(
        f"Running PyMC: step={args.pymc_step}, draws={args.steps}, tune={args.pymc_tune}, "
        f"chains={nchains}, mpi_ranks={size}."
    )
    args_bytes = cloudpickle.dumps(args)
    payloads = [(args_bytes, i) for i in range(nchains)]
    t0 = time()
    traces = pool.map(_run_pymc_chain, payloads)
    dt = time() - t0

    if len(traces) == 1:
        trace = traces[0]
    else:
        trace = az.concat(*traces, dim="chain")
    _add_attrs_to_idata(trace, _common_attrs(args, "pymc", runtime_seconds=dt))
    _write_idata(trace, idata_path)
    print(f"PyMC sampling took {_dt.timedelta(seconds=dt)}")
    print(f"Saved ArviZ InferenceData to {idata_path}")
    try:
        print(az.summary(trace, var_names=["theta"]))
    except Exception as exc:
        print(f"Could not print ArviZ summary: {exc}")


def _warmup_and_validate(args):
    if posterior.NDIM is None or int(posterior.NDIM) <= 0:
        raise ValueError(f"posterior.NDIM must be a positive integer; got {posterior.NDIM!r}.")
    sampler = args.sampler.lower()
    if sampler == "emcee":
        _emcee_imports()
        _require_emcee_args(args)
        p0 = _check_starting_location(args.chains)
        _check_logp(p0[0])
        return p0
    if sampler == "dynesty":
        _dynesty_imports()
        _check_for_log_likelihood()
        _check_for_prior_transform()
        theta0 = _check_starting_location(1)[0]
        ll = float(posterior.log_likelihood(theta0))
        if np.isnan(ll):
            raise ValueError("posterior.log_likelihood returned NaN at the warmup point.")
        _check_vector("posterior.prior_transform(0.5)", posterior.prior_transform(np.full(posterior.NDIM, 0.5)))
        return theta0
    if sampler == "pymc":
        _pymc_imports()
        _require_steps(args, "pymc")
        theta0 = _prior_mean_or_representative()
        _check_logp(theta0)
        return theta0
    raise ValueError(f"Unknown sampler {args.sampler}.")


def _serial_timing(args, warm_point):
    t0 = time()
    if args.sampler == "emcee":
        if hasattr(posterior, "log_posterior_batch"):
            posterior.log_posterior_batch(warm_point)
            label = f"{len(warm_point)} batched log_posterior calls"
        else:
            [posterior.log_posterior(theta) for theta in warm_point]
            label = f"{len(warm_point)} serial log_posterior calls"
    elif args.sampler == "dynesty":
        theta = posterior.prior_transform(np.full(posterior.NDIM, 0.5))
        posterior.log_likelihood(theta)
        label = "1 dynesty prior_transform + log_likelihood call"
    else:
        posterior.log_posterior(warm_point)
        label = "1 PyMC black-box log_posterior call"
    dt = time() - t0
    print(f"{label} in {_dt.timedelta(seconds=dt)} [hh:mm:ss]")


def _mpi_timing(args, pool, size):
    if size < 2:
        raise RuntimeError("--MPI-timing-test requires at least one MPI worker rank.")
    nworkers = size - 1
    t0 = time()
    if args.sampler == "emcee":
        base, rem = divmod(args.chains, nworkers)
        counts = [base + (i < rem) for i in range(nworkers)]
        inputs = [_check_starting_location(n) for n in counts if n > 0]
        func = posterior.log_posterior_batch if hasattr(posterior, "log_posterior_batch") else lambda xs: [posterior.log_posterior(x) for x in xs]
        pool.map(func, inputs)
        label = f"{sum(counts)} emcee log_posterior samples on {nworkers} workers"
    elif args.sampler == "dynesty":
        us = [np.random.default_rng(i).random(posterior.NDIM) for i in range(nworkers)]
        thetas = [posterior.prior_transform(u) for u in us]
        pool.map(posterior.log_likelihood, thetas)
        label = f"{nworkers} dynesty likelihood calls"
    else:
        thetas = [_pymc_initial_theta(args, i) for i in range(nworkers)]
        pool.map(posterior.log_posterior, thetas)
        label = f"{nworkers} PyMC log_posterior calls"
    dt = time() - t0
    print(f"{label} in {_dt.timedelta(seconds=dt)} [hh:mm:ss]")


def main(argv=None):
    global posterior
    args = parse_args(argv)
    posterior = importlib.import_module(args.posterior_module)
    posterior.init_posterior(args.input)
    warm_point = _warmup_and_validate(args)

    if args.serial_timing_test:
        _serial_timing(args, warm_point)
        return

    pool, size, using_mpi = _pool_context(args)
    with pool:
        if _is_worker(pool, using_mpi):
            pool.wait()
            return

        if args.MPI_timing_test:
            _mpi_timing(args, pool, size)
            return

        if args.sampler == "emcee":
            run_emcee(args, pool, size=size)
        elif args.sampler == "dynesty":
            run_dynesty(args, pool, size=size)
        elif args.sampler == "pymc":
            run_pymc(args, pool, size=size)
        else:
            raise ValueError(f"Unknown sampler {args.sampler}.")


if __name__ == "__main__":
    main()
