#!/bin/bash
#SBATCH --job-name=monitor
#SBATCH --partition=shared
#SBATCH --time=00-14:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
module load lang/Anaconda3/2023.03-1

conda info --envs
conda activate py39

for i in {1..12}
do
    ~/.conda/envs/py39/bin/python3.9 union3/scripts/setup_slurm_runs.py union3/scripts/union3 koa_scratch/Union3 100 0 1
    sleep 3600
done
