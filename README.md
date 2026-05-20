# 🅱️lack 🅱️ox 🅱️ayes

Do you want to do Bayesian inference with an expensive computer model? If your output space is small enough, you can probably emulate, so go check out [`surmise`](github.com/bandframework/surmise). Oh, you have a big output space? Well, if you model is differentiable, you probably want to use a fancy sampler that can take advantage of gradients, like [`PyMC`'s NUTS](https://www.pymc.io/projects/docs/en/v5.9.0/api/generated/pymc.NUTS.html).

Ahh, you have a big output space and your model is not differentiable? Welcome to Black Box Bayes! This package provides a simple CLI for running production-scale Bayesian inference on black-box models with `emcee`, `dynesty`, or `PyMC`. All you have to do is provide some minimal information (`log_posterior`, `log_prior`, etc.), and `black-box-bayes` (or 🅱️🅱️🅱️) can run production inference for your model using either

- [`emcee`](https://emcee.readthedocs.io/en/stable/) for affine-invariant ensemble sampling
- [`dynesty`](https://dynesty.readthedocs.io/en/latest/) for nested sampling and evidence estimation
- [`PyMC`](https://www.pymc.io/) for adaptive Metropolis-Hastings sampling

In all cases, the output is an ArviZ `InferenceData` NetCDF file, so post processing workflows are the same regardless of sampler choice. In all cases, massive parallelism is available via MPI (requiring `schwimmbad`), so high performance computing environments and many chains are no problem.

## What is 🅱️🅱️🅱️?

A small installable package that exposes a single CLI, `black-box-bayes`, for
black-box Bayesian inference with `emcee`, `dynesty`, or `PyMC`. Every sampler
writes an ArviZ `InferenceData` NetCDF file.

The driver expects `--input` to point at a pickled config-like object exposing:

```python
ndim
starting_location(nwalkers)
log_posterior(theta)
log_likelihood(theta)      # required for dynesty
prior_transform(u)         # required for dynesty
log_posterior_batch(thetas)  # optional
parameter_names           # optional
```

The object's defining module still needs to be importable when the pickle is
loaded (for example, by running from the model directory or putting that code on
`PYTHONPATH`).

## Local install

```bash
pip install -e .[emcee,test]
```

Install extra backends as needed:

```bash
pip install -e .[dynesty]
pip install -e .[pymc]
pip install -e .[all]
```

## Interface

```bash
black-box-bayes --help
```

prints:

```
usage: black-box-bayes [-h] --input INPUT [--output OUTPUT]
                       [--idata-results IDATA_RESULTS] [--sampler {emcee,dynesty,pymc}] [--no-mpi] [--require-mpi]
                       [--chains CHAINS] [--pymc-chains PYMC_CHAINS] [--steps STEPS] [--idata-discard IDATA_DISCARD]
                       [--idata-thin IDATA_THIN] [--emcee-backend EMCEE_BACKEND] [--burnin BURNIN]
                       [--batch-size BATCH_SIZE] [--step-size STEP_SIZE] [--rtol RTOL]
                       [--emcee-progress | --no-emcee-progress] [--serial-timing-test] [--MPI-timing-test]
                       [--dynesty-run {static,single,dynamic}] [--nlive NLIVE] [--nlive-batch NLIVE_BATCH]
                       [--dynesty-bound {none,single,multi,balls,cubes}]
                       [--dynesty-sample {auto,unif,rwalk,slice,rslice}] [--dynesty-walks DYNESTY_WALKS]
                       [--dynesty-slices DYNESTY_SLICES] [--dynesty-facc DYNESTY_FACC]
                       [--dynesty-bootstrap DYNESTY_BOOTSTRAP] [--dynesty-enlarge DYNESTY_ENLARGE]
                       [--dynesty-update-interval DYNESTY_UPDATE_INTERVAL] [--dlogz DLOGZ] [--dlogz-init DLOGZ_INIT]
                       [--maxiter MAXITER] [--maxcall MAXCALL] [--maxbatch MAXBATCH] [--n-effective N_EFFECTIVE]
                       [--dynesty-pfrac DYNESTY_PFRAC] [--dynesty-use-stop | --no-dynesty-use-stop]
                       [--add-live | --no-add-live] [--dynesty-progress | --no-dynesty-progress]
                       [--queue-size QUEUE_SIZE] [--dynesty-checkpoint DYNESTY_CHECKPOINT]
                       [--dynesty-checkpoint-every DYNESTY_CHECKPOINT_EVERY] [--dynesty-resume]
                       [--dynesty-results DYNESTY_RESULTS] [--dynesty-native-results DYNESTY_NATIVE_RESULTS]
                       [--no-dynesty-native-results] [--dynesty-history DYNESTY_HISTORY]
                       [--dynesty-equal-weight | --no-dynesty-equal-weight] [--seed SEED] [--pymc-tune PYMC_TUNE]
                       [--pymc-step {demetropolisz,demetropolis,metropolis}]
                       [--pymc-init {prior_mean,starting_location,random_prior}]
                       [--pymc-progress | --no-pymc-progress] [--pymc-target-accept PYMC_TARGET_ACCEPT]
                       [--pymc-results PYMC_RESULTS] [--pymc-random-seed PYMC_RANDOM_SEED]

Unified black-box Bayesian inference driver. This driver expects --input to point at a pickled
config-like object exposing: ndim starting_location(nwalkers) log_posterior(theta)
log_likelihood(theta) # required for dynesty prior_transform(u) # required for dynesty
log_posterior_batch(thetas) # optional; used for timing only parameter_names # optional
All samplers write an ArviZ InferenceData NetCDF file.

options:
  -h, --help            show this help message and exit
  --input INPUT         Path to pickled CalibrationConfig-like object.
  --output OUTPUT       Output directory.
  --idata-results IDATA_RESULTS
                        ArviZ InferenceData NetCDF output path.
  --sampler {emcee,dynesty,pymc}
  --no-mpi              Force serial execution even if MPI is installed.
  --require-mpi         Fail unless running with mpi4py/schwimmbad and at least one worker rank.
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

```


## Toy example

```bash
cd examples/toy
python make_config.py
black-box-bayes --input toy_config.pkl \
  --sampler emcee --chains 16 --steps 1000 --burnin 200 \
  --no-mpi --idata-results toy_emcee_idata.nc
```

Inspect the output:

```python
import arviz as az
idata = az.from_netcdf("toy_emcee_idata.nc")
print(idata.posterior["theta"].mean(("chain", "draw")))
```

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
