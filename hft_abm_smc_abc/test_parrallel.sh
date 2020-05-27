#!/bin/sh
#SBATCH --account=stats
#SBATCH --partition=ada
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --job-name="t=1Adaptive2Norm_2300"
#SBATCH --mail-user=gsnkel001@myuct.ac.za
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
module load python/anaconda-python-3.7
source activate RelevantName
python SMC_ABC.py 
