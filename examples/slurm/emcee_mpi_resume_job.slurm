#!/bin/bash --login

#SBATCH --account=<account_name>
#SBATCH --job-name=<job_name>
#SBATCH --ntasks=<number_of_tasks>
#SBATCH --cpus-per-task=1

# load modules, e.g.:
# module load python/3.8
# module load mpi

python -c 'import black_box_bayes; print("black-box-bayes is available")'

# Point this at the existing calibration directory that already contains
# config.pkl and chains.h5 from a previous emcee run.
CAL_DIR="./"
BACKEND_FILE="chains.h5"

if [[ ! -f "$CAL_DIR/$BACKEND_FILE" ]]; then
    echo "Expected existing emcee backend at $CAL_DIR/$BACKEND_FILE" >&2
    exit 1
fi

mpiexec -n "$SLURM_NTASKS" black-box-bayes \
    --input "$CAL_DIR/config.pkl" \
    --output "$CAL_DIR" \
    --sampler emcee \
    --require-mpi \
    --chains 256 \
    --steps 50000 \
    --batch-size 1000 \
    --step-size 2.0 \
    --burnin 0 \
    --emcee-backend "$BACKEND_FILE" \
    --idata-results "$CAL_DIR/emcee_idata_resumed.nc"

mv "${SLURM_JOB_NAME}-${SLURM_JOB_ID}.SLURMout" "${CAL_DIR}/slurm_output.log"
