#!/bin/bash

#SBATCH --job-name=sphexa
#SBATCH --output=sphexa-%j.out
#SBATCH --error=sphexa-%j.err

#SBATCH --gpus=rtx_4090:1 # 24 GB vram
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:20:00

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$REPO_ROOT/build"
EXECUTABLE="$BUILD_DIR/main/src/sphexa/sphexa-cuda"

module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 cuda/12.6.2 openmpi/4.1.7 hdf5/1.14.5
module list

make -C "$BUILD_DIR" -j sphexa-cuda

rm dump_*.h5

export OMP_NUM_THREADS=16
$EXECUTABLE --init sedov-magneto --prop magneto-ve --glass 50c.h5 -n 100 -s 1000 -w 10 -f x,y,z,rho,p

mkdir -p out/$SLURM_JOB_ID/
mv *.err *.out dump_*.h5 constants.txt profile.h5 out/$SLURM_JOB_ID/