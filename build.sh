#!/bin/bash

#SBATCH --job-name=sphexa-build
#SBATCH --output=sphexa-build-%j.out
#SBATCH --error=sphexa-build-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:10:00

module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 cuda/12.6.2 openmpi/4.1.7 hdf5/1.14.5
module list

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$REPO_ROOT/build"

make -C "$BUILD_DIR" -j