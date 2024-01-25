#!/bin/bash
#SBATCH --job-name=step2
#SBATCH --partition=shared
#SBATCH --time=0-12:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

cd /home/drubin/koa_scratch/sim_data_simplex/
python3.9 /home/drubin/union3//simulated/step1A_compareLC_vs_input.py read | tail

