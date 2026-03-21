#!/bin/bash

#SBATCH --job-name=sphexa-cpu
#SBATCH --output=logs/sphexa-cpu-%j.out
#SBATCH --error=logs/sphexa-cpu-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=1024
#SBATCH --time=04:00:00

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$REPO_ROOT/build/cpu"
EXECUTABLE="$BUILD_DIR/main/src/sphexa/sphexa"

module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 openmpi/4.1.7 hdf5/1.14.5
module list

make -C "$BUILD_DIR" -j sphexa

mkdir -p out/$SLURM_JOB_ID/

export OMP_NUM_THREADS=128

$EXECUTABLE \
    --init alfven-wave \
    --prop magneto-ve \
    --glass 50c.h5 \
    -n 100 \
    -s 1000 \
    -w 10 \
    -f x,y,z,rho,p,Bx,By,Bz
    -o out/$SLURM_JOB_ID/dump.h5 \
