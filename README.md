# 🅱️lack 🅱️ox 🅱️ayes

Do you want to do Bayesian inference with an expensive computer model? You can probably emulate it, so go check out [`surmise`](https://github.com/bandframework/surmise/)!

Oh, you have too big of an output space for easy emulation? Well, if your model is differentiable, you probably want to use a fancy sampler that can take advantage of gradients, like [`PyMC`'s NUTS](https://www.pymc.io/projects/docs/en/v5.9.0/api/generated/pymc.NUTS.html).

Ahh, you have a big output space and your model is not differentiable? That sucks, welcome to Black Box Bayes! This package provides a simple CLI for running production-scale Bayesian inference on black-box models with `emcee`, `ptemcee`, `dynesty`, `pocoMC`, or `PyMC`('s non-gradient-requiring) samplers. All you have to do is provide some minimal information (`log_posterior`, `log_prior`, etc.), and `black-box-bayes` (or 🅱️🅱️🅱️) can run production inference for your model using either

- [`emcee`](https://emcee.readthedocs.io/en/stable/) for affine-invariant ensemble sampling
- [`ptemcee`](https://github.com/NuclearTalent/ptemcee_2025) for adaptive parallel-tempered ensemble sampling (multi-modal posteriors and evidence estimation)
- [`dynesty`](https://dynesty.readthedocs.io/en/latest/) for nested sampling and evidence estimation
- [`pocoMC`](https://pocomc.readthedocs.io/en/latest/) for normalizing-flow-preconditioned sequential Monte Carlo (non-linear correlations, multi-modal posteriors, and evidence estimation)
- [`PyMC`](https://www.pymc.io/) for adaptive Metropolis-Hastings sampling

In all cases, the output is an ArviZ `InferenceData` NetCDF file, so post processing workflows are the same regardless of sampler choice. In all cases, massive parallelism is available via MPI (requiring `schwimmbad`), so high performance computing environments, many chains, and long calibrations are no problem.

If you publish results from 🅱️🅱️🅱️, please cite the sampler(s) you used; see [Citing](#citing).

## What is 🅱️🅱️🅱️?

A small installable package that exposes a single CLI, `black-box-bayes`, for
black-box Bayesian inference with `emcee`, `ptemcee`, `dynesty`, `pocoMC`, or `PyMC`.

The driver expects `--input` to point at a pickled config-like object exposing:

```python
ndim
starting_location(nwalkers)
log_posterior(theta)
log_likelihood(theta)      # required for dynesty, ptemcee, and pocomc
log_prior(theta)           # required for ptemcee and pocomc
prior_transform(u)         # required for dynesty and pocomc
log_posterior_batch(thetas)  # optional
parameter_names           # optional
prior_bounds              # optional; (ndim, 2) prior support for pocomc
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
pip install -e .[pocomc]   # pulls in torch
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
                   [--sampler {emcee,ptemcee,dynesty,pocomc,pymc}] [--pool {auto,serial,mpi,multiprocessing}]
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
                   [--dynesty-resume] [--dynesty-explore-batches DYNESTY_EXPLORE_BATCHES]
                   [--dynesty-explore-nlive DYNESTY_EXPLORE_NLIVE] [--dynesty-explore-maxcall DYNESTY_EXPLORE_MAXCALL]
                   [--dynesty-results DYNESTY_RESULTS] [--dynesty-native-results DYNESTY_NATIVE_RESULTS]
                   [--no-dynesty-native-results] [--dynesty-history DYNESTY_HISTORY]
                   [--dynesty-equal-weight | --no-dynesty-equal-weight] [--seed SEED]
                   [--pocomc-n-total POCOMC_N_TOTAL] [--pocomc-n-evidence POCOMC_N_EVIDENCE]
                   [--pocomc-n-effective POCOMC_N_EFFECTIVE] [--pocomc-n-active POCOMC_N_ACTIVE]
                   [--pocomc-n-steps POCOMC_N_STEPS] [--pocomc-n-max-steps POCOMC_N_MAX_STEPS]
                   [--pocomc-flow {nsf3,nsf6,nsf12,maf3,maf6,maf12}]
                   [--pocomc-precondition | --no-pocomc-precondition] [--pocomc-sample {tpcn,rwm}]
                   [--pocomc-pytorch-threads POCOMC_PYTORCH_THREADS] [--pocomc-progress | --no-pocomc-progress]
                   [--pocomc-checkpoint POCOMC_CHECKPOINT] [--pocomc-checkpoint-every POCOMC_CHECKPOINT_EVERY]
                   [--pocomc-resume] [--pocomc-native-results POCOMC_NATIVE_RESULTS] [--no-pocomc-native-results]
                   [--pocomc-equal-weight | --no-pocomc-equal-weight] [--pymc-tune PYMC_TUNE]
                   [--pymc-step {demetropolisz,demetropolis,metropolis}]
                   [--pymc-init {prior_mean,starting_location,random_prior}] [--pymc-progress | --no-pymc-progress]
                   [--pymc-target-accept PYMC_TARGET_ACCEPT] [--pymc-results PYMC_RESULTS]
                   [--pymc-random-seed PYMC_RANDOM_SEED]

Unified black-box Bayesian inference driver. This driver expects ``--input`` to point at a pickled config-like object
exposing: ndim starting_location(nwalkers) log_posterior(theta) log_likelihood(theta) # required for dynesty, ptemcee,
and pocomc log_prior(theta) # required for ptemcee and pocomc prior_transform(u) # required for dynesty and pocomc
log_posterior_batch(thetas) # optional; used for timing only parameter_names # optional prior_bounds # optional;
(ndim, 2) prior support for pocomc All samplers write an ArviZ InferenceData NetCDF file.

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to pickled CalibrationConfig-like object.
  --output OUTPUT       Output directory.
  --idata-results IDATA_RESULTS
                        ArviZ InferenceData NetCDF output path.
  --sampler {emcee,ptemcee,dynesty,pocomc,pymc}
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
  --dynesty-explore-batches DYNESTY_EXPLORE_BATCHES
                        Number of full-range (mode='full') batches to append after the main dynamic run. Full-range
                        batches re-sample the entire prior instead of the weight function's preferred likelihood
                        slice, which is dynesty's recommended way to discover modes a run has missed. Dynamic runs
                        only. --dynesty-use-stop does not govern these batches; they always run. Resume does not track
                        partial progress through them: a job that dies mid-exploration restarts the whole exploration
                        phase.
  --dynesty-explore-nlive DYNESTY_EXPLORE_NLIVE
                        Live points per exploration batch (default: --nlive-batch, else dynesty's default).
  --dynesty-explore-maxcall DYNESTY_EXPLORE_MAXCALL
                        Per-batch likelihood-call cap for exploration batches. A full-range batch redoes the whole
                        prior-to-posterior compression, so it is not otherwise bounded.
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
  --pocomc-n-total POCOMC_N_TOTAL
                        Convergence target: pocomc stops once the temperature reaches beta=1 and the importance-
                        weighted posterior has at least this effective sample size. Resuming a finished run with a
                        larger value keeps sampling.
  --pocomc-n-evidence POCOMC_N_EVIDENCE
                        Normalizing-flow importance samples for the evidence estimate (extra likelihood calls). 0 uses
                        the cheaper SMC estimate, which carries no error bar.
  --pocomc-n-effective POCOMC_N_EFFECTIVE
                        Target ESS per tempering step.
  --pocomc-n-active POCOMC_N_ACTIVE
                        Particles moved by MCMC per step; keep it at most --pocomc-n-effective and a multiple of the
                        worker count.
  --pocomc-n-steps POCOMC_N_STEPS
                        MCMC steps after logP plateaus (pocomc default: ndim//2).
  --pocomc-n-max-steps POCOMC_N_MAX_STEPS
                        Maximum MCMC steps per iteration (pocomc default: 10*n_steps).
  --pocomc-flow {nsf3,nsf6,nsf12,maf3,maf6,maf12}
  --pocomc-precondition, --no-pocomc-precondition
                        Precondition MCMC with a normalizing flow.
  --pocomc-sample {tpcn,rwm}
  --pocomc-pytorch-threads POCOMC_PYTORCH_THREADS
                        torch threads for flow training (master process only).
  --pocomc-progress, --no-pocomc-progress
  --pocomc-checkpoint POCOMC_CHECKPOINT
                        pocomc state file, overwritten in place (default: output/pocomc_checkpoint.state).
  --pocomc-checkpoint-every POCOMC_CHECKPOINT_EVERY
                        Minimum seconds between pocomc checkpoints; 0 checkpoints every iteration. The final state is
                        always written.
  --pocomc-resume       Resume from --pocomc-checkpoint when it exists.
  --pocomc-native-results POCOMC_NATIVE_RESULTS
                        pocomc-native particle-history archive (.npz). Defaults to output/pocomc_results.npz. This is
                        separate from the standardized ArviZ InferenceData output.
  --no-pocomc-native-results
                        Disable the extra pocomc-native .npz archive; ArviZ output is still written.
  --pocomc-equal-weight, --no-pocomc-equal-weight
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
| pocomc | `random_state=` (numpy's legacy global state and torch); prior draws use their own Generator |
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

Or with preconditioned Monte Carlo:

```bash
black-box-bayes --input toy_config.pkl \
  --sampler pocomc --pocomc-n-total 2048 \
  --no-mpi --idata-results toy_pocomc_idata.nc
```

Inspect the output:

```python
import arviz as az
idata = az.from_netcdf("toy_emcee_idata.nc")
print(idata.posterior["theta"].mean(("chain", "draw")))
```

## What is in the NetCDF

Every sampler writes the same posterior layout: a single variable `theta` with dimensions
`(chain, draw, theta_dim)`. The `theta_dim` coordinate is **labelled with your parameter
names**, so the file is self-describing — you do not need the config that produced it to
know which column is which:

```python
import arviz as az
idata = az.from_netcdf("toy_emcee_idata.nc")

print(idata.posterior["theta"].dims)                  # ('chain', 'draw', 'theta_dim')
print(idata.posterior.coords["theta_dim"].values)     # ['mu0' 'mu1']

# Select one parameter by name rather than by remembering its index:
print(idata.posterior["theta"].sel(theta_dim="mu0").mean())

# ArviZ labels its summary rows the same way: theta[mu0], theta[mu1].
print(az.summary(idata))
```

The names come from `parameter_names` (or `PARAMETER_NAMES` / `param_names`) on your config
object, in the same order as the columns of the `theta` vector passed to `log_posterior`.
They are recorded three ways, all of which survive the NetCDF round trip:

| where | value |
|---|---|
| `posterior.coords["theta_dim"]` | the ordered names — the authoritative source |
| `posterior.attrs["parameter_names"]` | comma-joined mirror, readable with `ncdump -h` |
| `posterior.attrs["parameter_names_source"]` | `config`, `generated`, or `generated_length_mismatch` |

If your config supplies no names, or supplies a number of them that disagrees with `ndim`,
the run falls back to generic `theta_0 … theta_n` labels and says so in
`parameter_names_source`. A count mismatch is also warned about at load time, before
sampling starts.

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

### Full-range exploration batches

In a dynamic run, every batch after the initial one is allocated by dynesty's *weight
function*, which targets the likelihood range where more points most improve the posterior
or the evidence (per `--dynesty-pfrac`). That is the right default, and it is exactly wrong
for **mode discovery**: the weight function concentrates effort where the current run says
it is valuable, which by construction cannot be a mode the run has not found.

`--dynesty-explore-batches N` appends N batches that re-sample the *entire* prior instead
(dynesty's `add_batch(mode='full')`, the FAQ's `logl_bounds=(-inf, inf)` recommendation).
Each one restarts the compression from the prior and gets an independent shot at every mode.

```bash
black-box-bayes --input cfg.pkl --sampler dynesty --dynesty-run dynamic \
  --nlive 1000 --dynesty-explore-batches 4 --dynesty-explore-nlive 500 \
  --dynesty-explore-maxcall 2000000
```

Reach for it when the posterior is multimodal, or when you suspect modes that separate only
at high likelihood. A full-range batch redoes the whole prior-to-posterior compression, so
it is not otherwise bounded — `--dynesty-explore-maxcall` is cheap insurance.

Three things worth knowing:

- **The evidence stays valid, but it can move a lot.** dynesty merges batches into one
  consistent dynamic-nested-sampling result, so `logz` remains a proper estimator. A jump
  much larger than its own error bar means the batch found mass the original run missed —
  that is the feature working, not a bug.
- **`logzerr` will not warn you.** The quoted error is the statistical error of the
  compression actually performed. It cannot see a mode that was never visited; in practice
  it can read ±0.3 on a value that is hundreds of nats wrong. The way to gain confidence is
  **batch-to-batch stability of `logz`**, printed after each batch, not a small `logzerr`.
- **Mine the dead points, not the posterior.** If you are using nested sampling to
  *catalogue* modes before building per-mode priors, the posterior is the wrong product — it
  correctly weights subdominant modes out of existence. The `samples`/`logl` arrays in the
  `.npz` are what you want; exploration batches are the cheapest way to enrich them.

Defaults to `0`, so existing command lines are unaffected. Requires `--dynesty-run dynamic`
(`NestedSampler` has no `add_batch`); combining it with a static run is an error rather than
a silently ignored flag. `--dynesty-use-stop` governs only the weight-function batches —
exploration batches run afterwards unconditionally. Each batch checkpoints, but resume does
not track partial progress through the exploration phase: a job that dies during batch 3 of
5 restarts all 5 on resume.

The run records `dynesty_explore_batches`, `dynesty_explore_nlive`, and
`dynesty_explore_maxcall` in the `.nc` attrs, so two results with different provenance stay
distinguishable.

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

## pocoMC

[pocoMC](https://pocomc.readthedocs.io) anneals a population of particles from the prior
to the posterior (sequential Monte Carlo), moving them with MCMC that is preconditioned
by a normalizing flow trained on the current particles. It handles strongly non-linear
correlations and multiple modes well and gives an evidence estimate, usually at far
fewer likelihood calls than nested sampling.

### Convergence

pocoMC has its own stopping rule, so `--steps`, `--rtol`, and `--batch-size` do not apply.
A run is done when the temperature reaches `beta = 1` **and** the importance-weighted
posterior has an effective sample size of at least `--pocomc-n-total` (default 4096). That
one flag is the convergence control: raise it for smoother posteriors. The evidence is then
estimated by importance sampling from the flow, using `--pocomc-n-evidence` extra likelihood
calls (0 uses the cheaper SMC estimate, which has no error bar). The run prints and records
the final ESS, iteration count, likelihood calls, and `logz ± err`.

Other knobs: `--pocomc-n-active` particles move per iteration (keep it a multiple of your
worker count, or workers sit idle), `--pocomc-n-effective` sets the ESS kept per tempering
step (pocoMC adapts it as it runs), and `--pocomc-flow`, `--no-pocomc-precondition`,
`--pocomc-sample`, `--pocomc-n-steps`, `--pocomc-n-max-steps` pass through.

### Checkpointing and resume

pocoMC's own scheme writes a new `pmc_<iteration>.state` file every N iterations, each
holding the whole particle history. Here pocoMC is driven so that it keeps **one**
checkpoint, `--pocomc-checkpoint` (default `output/pocomc_checkpoint.state`), rewritten
atomically at most every `--pocomc-checkpoint-every` seconds (default 300, like dynesty;
0 means every iteration). The final state is always written.

```bash
# the job died, or hit its wall time: pick up where the last checkpoint left off
black-box-bayes --input cfg.pkl --sampler pocomc --pocomc-resume ...

# the run finished but you want more posterior samples: resume with a larger target
black-box-bayes --input cfg.pkl --sampler pocomc --pocomc-resume --pocomc-n-total 16384 ...
```

Resuming a finished run with a larger `--pocomc-n-total` continues the annealed run
instead of starting over. Particle settings (`--pocomc-n-active`) come from the
checkpoint, and a mismatch is warned about. The resumed run uses whatever `--pool` you
launch it with.

### The prior

pocoMC draws its initial particles from `prior_transform` and uses `log_prior` inside its
MCMC target, so **the two must describe the same prior**, with `log_prior` returning
`-inf` outside the support. pocoMC also needs the prior's support, to map bounded parameters
to an unbounded space. That support is inferred from `prior_transform` one axis at a time,
which is exact for independent priors. For a correlated transform, expose
`prior_bounds` (shape `(ndim, 2)`, `±inf` for unbounded) on the config. Warmup checks prior
draws against the support and fails early if they disagree.

### Outputs

```text
pocomc_idata.nc          # ArviZ posterior view (equal-weight resample by default)
pocomc_results.npz       # full particle history: x, logl, logp, beta, logz, ... + evidence
pocomc_checkpoint.state  # resumable sampler state
```

In `pocomc_results.npz` every per-particle field is `(n_iter, n_active, ...)` and every
per-iteration field (`beta`, `logz`, `ess`, ...) is `(n_iter,)`. pocoMC keeps no
per-iteration weights: its `results` hands back one array of importance weights of all
stored particles *toward beta = 1*. That array is stored as `logw_posterior`, shaped like
`logl`, not as `logw`, so it cannot be mistaken for weights at each iteration's beta. For
the tempered posterior at any other beta, recompute the same persistent-sampling weights:

```python
import numpy as np
from black_box_bayes.cli import pocomc_log_weights

z = np.load("pocomc_results.npz")
logw = pocomc_log_weights(z["logl"], z["beta"], z["logz"], beta_final=0.3)  # shaped like logl
```

`--no-pocomc-equal-weight` keeps the weighted particles instead, with their
`importance_weight` in `sample_stats`. Under MPI or multiprocessing only likelihood calls
are distributed; flow training runs in the master process (`--pocomc-pytorch-threads`).

## Tests

```bash
pip install -e .[test]
pytest
```

The pocoMC integration tests are skipped unless `pocomc` is installed (`pip install -e .[test,pocomc]`).

To run the optional MPI integration coverage as well, install the MPI extras and
use an environment that provides `mpiexec`:

```bash
pip install -e .[test,mpi]
pytest tests/test_toy_cli.py -k mpi
```

## Citing

🅱️🅱️🅱️ is a thin wrapper: all the actual sampling is done by the packages below.
If you use 🅱️🅱️🅱️ in published work, **please cite the sampler(s) you actually ran**,
along with the papers describing the algorithms they implement. The entries below
are the ones each project asks for.

### emcee (`--sampler emcee`)

```bibtex
@article{ForemanMackey2013,
  author  = {Foreman-Mackey, Daniel and Hogg, David W. and Lang, Dustin and Goodman, Jonathan},
  title   = {emcee: The MCMC Hammer},
  journal = {Publications of the Astronomical Society of the Pacific},
  volume  = {125},
  number  = {925},
  pages   = {306--312},
  year    = {2013},
  doi     = {10.1086/670067}
}

@article{Goodman2010,
  author  = {Goodman, Jonathan and Weare, Jonathan},
  title   = {Ensemble samplers with affine invariance},
  journal = {Communications in Applied Mathematics and Computational Science},
  volume  = {5},
 number  = {1},
  pages   = {65--80},
  year    = {2010},
  doi     = {10.2140/camcos.2010.5.65}
}
```

### ptemcee (`--sampler ptemcee`)

The `ptemcee` README asks for both the adaptive parallel-tempering paper and the
`emcee` paper (`ForemanMackey2013` above), since `ptemcee` is built on `emcee`.

```bibtex
@article{Vousden2016,
  author  = {Vousden, W. D. and Farr, W. M. and Mandel, I.},
  title   = {Dynamic temperature selection for parallel tempering in Markov chain Monte Carlo simulations},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume  = {455},
  number  = {2},
  pages   = {1919--1937},
  year    = {2016},
  doi     = {10.1093/mnras/stv2422}
}
```

### dynesty (`--sampler dynesty`)

Cite the `dynesty` paper and Skilling's nested sampling papers:

```bibtex
@article{Speagle2020,
  author  = {Speagle, Joshua S.},
  title   = {dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume  = {493},
  number  = {3},
  pages   = {3132--3158},
  year    = {2020},
  doi     = {10.1093/mnras/staa278}
}


@inproceedings{Skilling2004,
  author    = {Skilling, John},
  title     = {Nested Sampling},
  booktitle = {AIP Conference Proceedings},
  volume    = {735},
  pages     = {395--405},
  year      = {2004},
  doi       = {10.1063/1.1835238}
}

@article{Skilling2006,
  author  = {Skilling, John},
  title   = {Nested sampling for general Bayesian computation},
  journal = {Bayesian Analysis},
  volume  = {1},
  number  = {4},
  pages   = {833--859},
  year    = {2006},
  doi     = {10.1214/06-BA127}
}
```

Depending on the options you chose, `dynesty` also asks you to cite the papers
behind them, e.g. Higson et al. (2019,
[doi:10.1007/s11222-018-9844-0](https://doi.org/10.1007/s11222-018-9844-0)) for
`--dynesty-run dynamic`, Feroz, Hobson & Bridges (2009,
[doi:10.1111/j.1365-2966.2009.14548.x](https://doi.org/10.1111/j.1365-2966.2009.14548.x))
for `--dynesty-bound multi`, and the slice-sampling papers for `slice`/`rslice`.
See the [dynesty citation guide](https://dynesty.readthedocs.io/en/latest/references.html)
for the full list. You can also print `sampler.citations` on the native `dynesty`
sampler to get the references for your exact configuration.

### pocoMC (`--sampler pocomc`)

```bibtex
@article{Karamanis2022MNRAS,
  author  = {Karamanis, Minas and Beutler, Florian and Peacock, John A. and Nabergoj, David and Seljak, Uro{\v{s}}},
  title   = {Accelerating astronomical and cosmological inference with preconditioned Monte Carlo},
  journal = {Monthly Notices of the Royal Astronomical Society},
  volume  = {516},
  number  = {2},
  pages   = {1644--1653},
  year    = {2022},
  doi     = {10.1093/mnras/stac2272}
}

@article{Karamanis2022JOSS,
  author  = {Karamanis, Minas and Nabergoj, David and Beutler, Florian and Peacock, John A. and Seljak, Uro{\v{s}}},
  title   = {pocoMC: A Python package for accelerated Bayesian inference in astronomy and cosmology},
  journal = {Journal of Open Source Software},
  volume  = {7},
  number  = {79},
  pages   = {4634},
  year    = {2022},
  doi     = {10.21105/joss.04634}
}
```

### PyMC (`--sampler pymc`)

Cite PyMC, plus the paper for the step method you selected with `--pymc-step`:
ter Braak & Vrugt (2008) for `demetropolisz` (the default) or ter Braak (2006) for
`demetropolis`.

```bibtex
@article{AbrilPla2023,
  author  = {Abril-Pla, Oriol and Andreani, Virgile and Carroll, Colin and Dong, Larry and Fonnesbeck, Christopher J. and Kochurov, Maxim and Kumar, Ravin and Lao, Junpeng and Luhmann, Christian C. and Martin, Osvaldo A. and Osthege, Michael and Vieira, Ricardo and Wiecki, Thomas and Zinkov, Robert},
  title   = {PyMC: a modern, and comprehensive probabilistic programming framework in Python},
  journal = {PeerJ Computer Science},
  volume  = {9},
  pages   = {e1516},
  year    = {2023},
  doi     = {10.7717/peerj-cs.1516}
}

@article{terBraak2008,
  author  = {ter Braak, Cajo J. F. and Vrugt, Jasper A.},
  title   = {Differential Evolution Markov Chain with snooker updater and fewer chains},
  journal = {Statistics and Computing},
  volume  = {18},
  number  = {4},
  pages   = {435--446},
  year    = {2008},
  doi     = {10.1007/s11222-008-9104-9}
}

@article{terBraak2006,
  author  = {ter Braak, Cajo J. F.},
  title   = {A Markov Chain Monte Carlo version of the genetic algorithm Differential Evolution: easy Bayesian computing for real parameter spaces},
  journal = {Statistics and Computing},
  volume  = {16},
  number  = {3},
  pages   = {239--249},
  year    = {2006},
  doi     = {10.1007/s11222-006-8769-1}
}
```
