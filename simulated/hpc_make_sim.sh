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


sim_sel_effects = int(sys.argv[1])
fit_sel_effects = int(sys.argv[2])
sim_sig_int = int(sys.argv[3])
sim_two_beta = int(sys.argv[4])
fit_two_beta = int(sys.argv[5])
k_correct = int(sys.argv[6])
