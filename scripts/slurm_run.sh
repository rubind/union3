#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=2-0:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
cd /home/drubin/nfs_fs02/union3/scripts/union3_fast
python ../read_and_sample.py input*pickle 2 > log.txt

