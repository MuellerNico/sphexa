#!/bin/bash

#SBATCH --job-name=sphexa     # Job name    (default: sbatch)
#SBATCH --output=sphexa-%j.out # Output file (default: slurm-%j.out)
#SBATCH --error=sphexa-%j.err  # Error file  (default: slurm-%j.out)

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2048
#SBATCH --time=00:10:00           # Wall clock time limit

module load stack/.2025-06-silent stack/2025-06
module load gcc/12.2.0 cmake/3.30.5 cuda/12.6.2 openmpi/4.1.7 hdf5/1.14.5
module list

echo "building..."
cd build
make -j sphexa
cd .. # back to root
mv build/main/src/sphexa/sphexa ./
echo "build done."

echo "Checking file existence..."
ls -lh /cluster/home/nicolmueller/sphexa/50c.h5 || echo "FILE NOT FOUND"
pwd

export OMP_NUM_THREADS=16
rm dump_*.h5
./sphexa --init sedov --glass /cluster/home/nicolmueller/sphexa/50c.h5 -n 100 -s 1000 -w 10 -f x,y,z,rho,p