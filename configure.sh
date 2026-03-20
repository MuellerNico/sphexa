#!/bin/bash
module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 openmpi/4.1.7 hdf5/1.14.5 cuda/12.6.2
module list

REPO_ROOT=$(git rev-parse --show-toplevel)

cmake --fresh -S "$REPO_ROOT" -B "$REPO_ROOT/build/gpu" -DCMAKE_CUDA_ARCHITECTURES="89"
# cmake --fresh -S "$REPO_ROOT" -B "$REPO_ROOT/build/cpu"