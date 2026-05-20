#!/bin/bash --login

#SBATCH --account=<account_name>
#SBATCH --job-name=<job_name>
#SBATCH --ntasks=<number_of_tasks>
#SBATCH --cpus-per-task=1

# load modules, e.g.:
# module load python/3.8
# module load mpi 

python -c 'import black_box_bayes; print("black-box-bayes is available")'

CAL_DIR="./"

mpiexec -n "$SLURM_NTASKS" black-box-bayes \
    --input "$CAL_DIR/config.pkl" \
    --output "$CAL_DIR" \
    --sampler emcee \
    --require-mpi \
    --chains 256 \
    --steps 50000 \
    --batch-size 1000 \
    --step-size 2.0 \
    --emcee-backend chains.h5 \
    --idata-results "$CAL_DIR/emcee_idata.nc"

mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.SLURMout" "${CAL_DIR}/slurm_output.log"
