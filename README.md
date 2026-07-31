# 🅱️lack 🅱️ox 🅱️ayes

Do you want to do Bayesian inference with an expensive computer model? You can probably emulate it, so go check out [`surmise`](https://github.com/bandframework/surmise/)!

Oh, you have too big of an output space for easy emulation? Well, if your model is differentiable, you probably want to use a fancy sampler that can take advantage of gradients, like [`PyMC`'s NUTS](https://www.pymc.io/projects/docs/en/v5.9.0/api/generated/pymc.NUTS.html).

Ahh, you have a big output space and your model is not differentiable? That sucks, welcome to Black Box Bayes! This package provides a simple CLI for running production-scale Bayesian inference on black-box models with `emcee`, `ptemcee`, `dynesty`, or `PyMC`('s non-gradient-requiring) samplers. All you have to do is provide some minimal information (`log_posterior`, `log_prior`, etc.), and `black-box-bayes` (or 🅱️🅱️🅱️) can run production inference for your model using either

- [`emcee`](https://emcee.readthedocs.io/en/stable/) for affine-invariant ensemble sampling
- [`ptemcee`](https://github.com/NuclearTalent/ptemcee_2025) for adaptive parallel-tempered ensemble sampling (multi-modal posteriors and evidence estimation)
- [`dynesty`](https://dynesty.readthedocs.io/en/latest/) for nested sampling and evidence estimation
- [`PyMC`](https://www.pymc.io/) for adaptive Metropolis-Hastings sampling

In all cases, the output is an ArviZ `InferenceData` NetCDF file, so post processing workflows are the same regardless of sampler choice. In all cases, massive parallelism is available via MPI (requiring `schwimmbad`), so high performance computing environments, many chains, and long calibrations are no problem.

## What is 🅱️🅱️🅱️?

A small installable package that exposes a single CLI, `black-box-bayes`, for
black-box Bayesian inference with `emcee`, `ptemcee`, `dynesty`, or `PyMC`.

The driver expects `--input` to point at a pickled config-like object exposing:

```python
ndim
starting_location(nwalkers)
log_posterior(theta)
log_likelihood(theta)      # required for dynesty and ptemcee
log_prior(theta)           # required for ptemcee
prior_transform(u)         # required for dynesty
log_posterior_batch(thetas)  # optional
parameter_names           # optional
```

`ptemcee` tempers the likelihood while holding the prior fixed, so it needs
`log_likelihood` and `log_prior` as two separate callables (unlike `emcee`, which
only needs the combined `log_posterior`).

The object's defining module still needs to be importable when the pickle is
loaded (for example, by running from the model directory or putting that code on
`PYTHONPATH`).

## Local install

```bash
pip install -e .[emcee,test]
```

Install extra backends as needed:

```bash
pip install -e .[ptemcee]
pip install -e .[dynesty]
pip install -e .[pymc]
pip install -e .[all]
```

The `ptemcee` extra installs the v2 (attrs-based) `ptemcee` API from PyPI. To use
the NuclearTalent fork instead:

```bash
pip install "git+https://github.com/NuclearTalent/ptemcee_2025"
```

## Interface

```bash
black-box-bayes --help
```

prints:

```
usage: __main__.py [-h] --input INPUT [--output OUTPUT] [--idata-results IDATA_RESULTS]
                   [--sampler {emcee,ptemcee,dynesty,pymc}] [--pool {auto,serial,mpi,multiprocessing}]
                   [--nprocs NPROCS] [--no-mpi] [--require-mpi] [--chains CHAINS] [--pymc-chains PYMC_CHAINS]
                   [--steps STEPS] [--idata-discard IDATA_DISCARD] [--idata-thin IDATA_THIN]
                   [--emcee-backend EMCEE_BACKEND] [--burnin BURNIN] [--batch-size BATCH_SIZE] [--step-size STEP_SIZE]
                   [--rtol RTOL] [--emcee-progress | --no-emcee-progress] [--ptemcee-ntemps PTEMCEE_NTEMPS]
                   [--ptemcee-tmax PTEMCEE_TMAX] [--ptemcee-adaptive | --no-ptemcee-adaptive]
                   [--ptemcee-progress | --no-ptemcee-progress] [--ptemcee-native-results PTEMCEE_NATIVE_RESULTS]
                   [--no-ptemcee-native-results] [--serial-timing-test] [--MPI-timing-test]
                   [--dynesty-run {static,single,dynamic}] [--nlive NLIVE] [--nlive-batch NLIVE_BATCH]
                   [--dynesty-bound {none,single,multi,balls,cubes}] [--dynesty-sample {auto,unif,rwalk,slice,rslice}]
                   [--dynesty-walks DYNESTY_WALKS] [--dynesty-slices DYNESTY_SLICES] [--dynesty-facc DYNESTY_FACC]
                   [--dynesty-bootstrap DYNESTY_BOOTSTRAP] [--dynesty-enlarge DYNESTY_ENLARGE]
                   [--dynesty-update-interval DYNESTY_UPDATE_INTERVAL] [--dlogz DLOGZ] [--dlogz-init DLOGZ_INIT]
                   [--maxiter MAXITER] [--maxcall MAXCALL] [--maxbatch MAXBATCH] [--n-effective N_EFFECTIVE]
                   [--dynesty-pfrac DYNESTY_PFRAC] [--dynesty-use-stop | --no-dynesty-use-stop]
                   [--add-live | --no-add-live] [--dynesty-progress | --no-dynesty-progress] [--queue-size QUEUE_SIZE]
                   [--dynesty-checkpoint DYNESTY_CHECKPOINT] [--dynesty-checkpoint-every DYNESTY_CHECKPOINT_EVERY]
                   [--dynesty-resume] [--dynesty-results DYNESTY_RESULTS]
                   [--dynesty-native-results DYNESTY_NATIVE_RESULTS] [--no-dynesty-native-results]
                   [--dynesty-history DYNESTY_HISTORY] [--dynesty-equal-weight | --no-dynesty-equal-weight]
                   [--seed SEED] [--pymc-tune PYMC_TUNE] [--pymc-step {demetropolisz,demetropolis,metropolis}]
                   [--pymc-init {prior_mean,starting_location,random_prior}] [--pymc-progress | --no-pymc-progress]
                   [--pymc-target-accept PYMC_TARGET_ACCEPT] [--pymc-results PYMC_RESULTS]
                   [--pymc-random-seed PYMC_RANDOM_SEED]

Unified black-box Bayesian inference driver. This driver expects ``--input`` to point at a pickled config-like object
exposing: ndim starting_location(nwalkers) log_posterior(theta) log_likelihood(theta) # required for dynesty and
ptemcee log_prior(theta) # required for ptemcee prior_transform(u) # required for dynesty log_posterior_batch(thetas)
# optional; used for timing only parameter_names # optional All samplers write an ArviZ InferenceData NetCDF file.

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to pickled CalibrationConfig-like object.
  --output OUTPUT       Output directory.
  --idata-results IDATA_RESULTS
                        ArviZ InferenceData NetCDF output path.
  --sampler {emcee,ptemcee,dynesty,pymc}
  --pool {auto,serial,mpi,multiprocessing}
                        Parallel execution mode. 'auto' (default) uses MPI when running under a multi-rank launcher
                        and falls back to serial. 'multiprocessing' spreads likelihood evaluations over local cores
                        without MPI -- use this on a laptop.
  --nprocs NPROCS       Worker processes for --pool multiprocessing (default: os.cpu_count()).
  --no-mpi              Force serial execution even if MPI is installed. Equivalent to --pool serial.
  --require-mpi         Fail unless running with mpi4py/schwimmbad and at least one worker rank. Equivalent to --pool
                        mpi.
  --chains CHAINS       emcee walkers or PyMC chains.
  --pymc-chains PYMC_CHAINS
                        PyMC chains, overriding --chains.
  --steps STEPS         MCMC draws/steps for emcee or PyMC.
  --idata-discard IDATA_DISCARD
                        Discard this many emcee draws when exporting InferenceData.
  --idata-thin IDATA_THIN
                        Thin emcee draws by this factor when exporting InferenceData.
  --emcee-backend EMCEE_BACKEND
                        emcee HDF5 backend filename inside --output.
  --burnin BURNIN
  --batch-size BATCH_SIZE
  --step-size STEP_SIZE
  --rtol RTOL
  --emcee-progress, --no-emcee-progress
  --ptemcee-ntemps PTEMCEE_NTEMPS
                        Number of ptemcee temperatures in the ladder.
  --ptemcee-tmax PTEMCEE_TMAX
                        Maximum ptemcee ladder temperature (e.g. inf for adaptive PT); default derives Tmax from
                        --ptemcee-ntemps.
  --ptemcee-adaptive, --no-ptemcee-adaptive
                        Enable adaptive parallel tempering.
  --ptemcee-progress, --no-ptemcee-progress
  --ptemcee-native-results PTEMCEE_NATIVE_RESULTS
                        ptemcee-native all-temperature results archive (.npz). Defaults to output/ptemcee_results.npz.
                        This is separate from the standardized ArviZ InferenceData output, which holds only the cold
                        posterior chain.
  --no-ptemcee-native-results
                        Disable the extra ptemcee-native .npz archive; ArviZ output is still written.
  --serial-timing-test
  --MPI-timing-test
  --dynesty-run {static,single,dynamic}
  --nlive NLIVE
  --nlive-batch NLIVE_BATCH
  --dynesty-bound {none,single,multi,balls,cubes}
  --dynesty-sample {auto,unif,rwalk,slice,rslice}
  --dynesty-walks DYNESTY_WALKS
  --dynesty-slices DYNESTY_SLICES
  --dynesty-facc DYNESTY_FACC
  --dynesty-bootstrap DYNESTY_BOOTSTRAP
  --dynesty-enlarge DYNESTY_ENLARGE
  --dynesty-update-interval DYNESTY_UPDATE_INTERVAL
  --dlogz DLOGZ
  --dlogz-init DLOGZ_INIT
  --maxiter MAXITER
  --maxcall MAXCALL
  --maxbatch MAXBATCH
  --n-effective N_EFFECTIVE
  --dynesty-pfrac DYNESTY_PFRAC
  --dynesty-use-stop, --no-dynesty-use-stop
  --add-live, --no-add-live
  --dynesty-progress, --no-dynesty-progress
  --queue-size QUEUE_SIZE
  --dynesty-checkpoint DYNESTY_CHECKPOINT
  --dynesty-checkpoint-every DYNESTY_CHECKPOINT_EVERY
  --dynesty-resume
  --dynesty-results DYNESTY_RESULTS
                        Deprecated alias for --idata-results in dynesty mode.
  --dynesty-native-results DYNESTY_NATIVE_RESULTS
                        Dynesty-native weighted results archive (.npz). Defaults to output/dynesty_results.npz for
                        dynesty runs. This is separate from the standardized ArviZ InferenceData output.
  --no-dynesty-native-results
                        Disable the extra dynesty-native .npz archive; ArviZ output is still written.
  --dynesty-history DYNESTY_HISTORY
                        Dynesty evaluation-history file; use 'none' to disable.
  --dynesty-equal-weight, --no-dynesty-equal-weight
  --seed SEED
  --pymc-tune PYMC_TUNE
  --pymc-step {demetropolisz,demetropolis,metropolis}
  --pymc-init {prior_mean,starting_location,random_prior}
  --pymc-progress, --no-pymc-progress
  --pymc-target-accept PYMC_TARGET_ACCEPT
  --pymc-results PYMC_RESULTS
                        Deprecated alias for --idata-results in PyMC mode.
  --pymc-random-seed PYMC_RANDOM_SEED
                        Base seed for PyMC chains (chain i gets seed+i). Defaults to --seed.
```


## Parallelism

`--pool` selects how likelihood evaluations are spread. The default, `auto`, uses MPI when
launched under a multi-rank launcher and silently falls back to serial otherwise.

```bash
# laptop: N local worker processes, no MPI needed
black-box-bayes --input cfg.pkl --pool multiprocessing --nprocs 8 ...

# cluster: one master rank plus N-1 workers
mpiexec -n 64 black-box-bayes --input cfg.pkl --pool mpi ...
```

`--no-mpi` and `--require-mpi` still work and are equivalent to `--pool serial` and
`--pool mpi`.

Two things differ between the MPI and multiprocessing paths:

- **Worker accounting.** Under MPI, rank 0 coordinates and does not evaluate, so `N` ranks
  give `N-1` workers. A multiprocessing pool has no master, so `--nprocs N` gives `N`
  workers. dynesty's `--queue-size` defaults accordingly; override it explicitly if you
  want a specific value.
- **Worker startup.** On Linux the pool forks after the config is loaded, so workers
  inherit an already-built model for free. On macOS and Windows Python spawns instead, and
  each worker reconstructs the config independently — for an expensive forward model that
  is a real per-worker startup cost, paid concurrently. Configs must be picklable either
  way.

If your forward model is itself threaded, pin it to one thread per worker
(`OMP_NUM_THREADS=1`, `NUMBA_NUM_THREADS=1`) or the processes will oversubscribe the
machine. With numba specifically, live threads plus `fork` is the one combination that can
hang outright.

## Reproducibility

`--seed` makes a run reproducible for every sampler. It has to be applied in four
different places because each backend sources its randomness differently:

| sampler | mechanism |
|---|---|
| dynesty | `rstate=` Generator |
| ptemcee | `random=` RandomState threaded through burn-in and production |
| emcee | internal RandomState, seeded from numpy's legacy global state at construction |
| pymc | `random_seed=`, chain `i` gets `seed + i`; override with `--pymc-random-seed` |

Two caveats worth knowing:

- **Starting positions are the config's business, not `--seed`'s.** They come from your
  `starting_location()`, so a config carrying its own RNG governs them regardless.
- **dynesty results depend on `--queue-size`.** The queue size sets how many points are
  proposed per iteration and therefore the order in which the RNG is consumed, so the same
  seed at a different worker count gives a different — equally valid — run. Fix
  `--queue-size` as well as `--seed` if you need runs to match across machines.

## Toy example

```bash
cd examples/toy
python make_config.py
black-box-bayes --input toy_config.pkl \
  --sampler emcee --chains 16 --steps 1000 --burnin 200 \
  --no-mpi --idata-results toy_emcee_idata.nc
```

Or with parallel tempering:

```bash
black-box-bayes --input toy_config.pkl \
  --sampler ptemcee --chains 16 --steps 1000 --ptemcee-ntemps 8 \
  --no-mpi --idata-results toy_ptemcee_idata.nc
```

Inspect the output:

```python
import arviz as az
idata = az.from_netcdf("toy_emcee_idata.nc")
print(idata.posterior["theta"].mean(("chain", "draw")))
```

## Worked example: comparing samplers and computing evidence

[`examples/alpha_ca48/`](examples/alpha_ca48/) is a full physics calibration — α elastic
scattering on ⁴⁸Ca at 28.2 MeV, using [jitr](https://github.com/beykyle/jitr) as the
forward model — run as a **4 sampler × 2 model matrix**.

It exists to show two things the toy example cannot:

- **Evidence.** `dynesty` and `ptemcee` each produce a marginal likelihood by completely
  different routes, so running both on two competing optical-model parameterizations
  gives a Bayes factor *and* a cross-check on the estimators. `emcee` and `pymc` produce
  no evidence at all.
- **Sampler behaviour on a hard posterior.** The real-potential depth is subject to the
  classic optical-model discrete ambiguity — well-separated basins that ensemble
  samplers can struggle to cross. How much of that structure each sampler resolves is
  something the example measures rather than assumes.

Requires the `examples` extra (Python ≥ 3.12) and runs on a laptop in an hour or so.
See its [README](examples/alpha_ca48/README.md).

## Dynesty outputs

For all samplers, the common downstream output is an ArviZ NetCDF file:

```bash
--idata-results run_idata.nc
```

Dynesty is different from MCMC samplers because its native result is a weighted nested-sampling record rather than equal-weight `chain x draw` posterior samples. For dynesty runs, `black-box-bayes` therefore writes both:

```text
dynesty_idata.nc       # ArviZ-compatible equal-weight posterior view
dynesty_results.npz    # dynesty-native weighted result archive
```

Control the native archive with:

```bash
--dynesty-native-results path/to/dynesty_results.npz
--no-dynesty-native-results
```

The `.npz` archive stores array-like fields such as `samples`, `logl`, `logwt`, `logz`, `logzerr`, `logvol`, `ncall`, and any other portable array fields exposed by dynesty's `Results.asdict()`.

## ptemcee outputs

`ptemcee` runs an ensemble at multiple temperatures. Only the cold chain (inverse temperature `beta = 1`) is the target posterior, so the standard ArviZ NetCDF holds just that cold chain (walkers mapped to ArviZ chains, exactly like `emcee`). The full parallel-tempered representation and the thermodynamic-integration log-evidence estimate are preserved in a separate native `.npz` archive:

```text
ptemcee_idata.nc       # ArviZ cold-chain posterior view
ptemcee_results.npz    # all-temperature chains + evidence estimate
```

Control the temperature ladder and native archive with:

```bash
--ptemcee-ntemps 8                     # number of temperatures
--ptemcee-tmax inf                     # optional max temperature (inf recommended for adaptive PT)
--no-ptemcee-adaptive                  # disable adaptive tempering (on by default)
--ptemcee-native-results path/to/ptemcee_results.npz
--no-ptemcee-native-results
```

The `.npz` archive stores `x` (shape `(draw, ntemps, walker, dim)`), the tempered `logl` and `logP` traces, the `betas` ladder, and the `log_evidence`/`log_evidence_err` estimate. A `ptemcee` run reuses the shared `--chains` (walkers), `--steps`, `--burnin`, `--batch-size`, `--rtol`, `--step-size` (proposal scale factor), and `--idata-discard`/`--idata-thin` flags.

## Tests

```bash
pip install -e .[test]
pytest
```

To run the optional MPI integration coverage as well, install the MPI extras and
use an environment that provides `mpiexec`:

```bash
pip install -e .[test,mpi]
pytest tests/test_toy_cli.py -k mpi
```
