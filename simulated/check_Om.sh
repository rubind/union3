#!/bin/bash
#SBATCH --job-name=checkruns
#SBATCH --partition=shared
#SBATCH --time=0-18:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

for i in {1..72}
do
    cd /home/drubin/koa_scratch/sim_data_simplex/
    python /home/drubin/union3//simulated/step1B_Om_sampled_well.py 1
    
    sleep 900
done
