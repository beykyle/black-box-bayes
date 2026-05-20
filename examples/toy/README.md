# Toy Gaussian example

From this directory after installing the package:

```bash
python make_config.py
black-box-bayes --input toy_config.pkl \
  --sampler emcee --chains 16 --steps 1000 --burnin 200 \
  --no-mpi --idata-results toy_emcee_idata.nc
```

Inspect the result:

```python
import arviz as az
idata = az.from_netcdf("toy_emcee_idata.nc")
print(idata.posterior["theta"].mean(("chain", "draw")))
```
