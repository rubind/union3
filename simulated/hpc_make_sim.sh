#!/bin/bash
#SBATCH --job-name=unity
#SBATCH --partition=shared
#SBATCH --time=1-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
cd /home/drubin/nfs_fs02/simulated_data_runs_no_twobeta/
python $UNITY/simulated/make_sim_multidataset.py 1 1 1 0 0 1 > log_sim.txt
