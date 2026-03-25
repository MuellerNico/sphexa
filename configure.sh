#!/bin/bash
module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 openmpi/4.1.7 hdf5/1.14.5 cuda/12.6.2
module list

REPO_ROOT=$(git rev-parse --show-toplevel)
BUILD_DIR="$REPO_ROOT/build"

rm -rf "$BUILD_DIR/cpu" "$BUILD_DIR/gpu"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR/cpu"
CC=mpicc CXX=mpicxx cmake \
    -S "$REPO_ROOT" \
    -B "$BUILD_DIR/gpu" \
    -DCSTONE_WITH_GPU_AWARE_MPI=OFF \
    -DCMAKE_CUDA_FLAGS=-ccbin=mpicxx \
    -DCMAKE_CUDA_ARCHITECTURES="89;80" # rtx 4090