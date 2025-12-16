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


"""orig_dir = sys.argv[1]
new_dir_loc = sys.argv[2]
n_copy = int(sys.argv[3])
cosmomodels = [int(item) for item in sys.argv[4]]
redo_runs = int(sys.argv[5])
leave_running_jobs_alone = int(sys.argv[6])
max_params_to_save = int(sys.argv[7])
"""

cd /home/drubin/koa_scratch/Union31_18_runs
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3_UNITY1.8_template ./ 10 15 0 1  100
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY_beta_B_fast_slow ./ 10 15 0 1 100
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY_beta_R_fast_slow ./ 10 15 0 1 100
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY_three_mode ./ 10 15 0 1 100
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY1.8_indivM ./ 20 1 0 1 100

~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY1.8_template ./ 100 1 0 1 10000
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY1.8_template ./	100 5 0	1 1000
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY1.8_template ./ 100 2 0 1 100
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/setup_slurm_runs.py  Union31_runs/union3.1_UNITY1.8_template ./ 20 3 0 1 100

