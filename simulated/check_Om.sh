#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=0-12:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

for i in {1..24}
do
    cd /home/drubin/koa_scratch/sim_data_simplex/
    python /home/drubin/union3//simulated/step1B_Om_sampled_well.py 1
    
    sleep 60
    
    cd /home/drubin/koa_scratch/sim_data_simplex_20/
    python /home/drubin/union3//simulated/step1B_Om_sampled_well.py 1

    sleep 1800
done
