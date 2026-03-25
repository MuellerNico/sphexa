#!/bin/bash

#SBATCH --job-name=sphexa-plot
#SBATCH --output=logs/sphexa-plot-%j.out
#SBATCH --error=logs/sphexa-plot-%j.err

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4096
#SBATCH --time=00:30:00

module load stack/.2025-06-silent stack/2025-06
module load python
module list

python scripts/plot_density_slice.py out/61147760/dump.h5 0-99
