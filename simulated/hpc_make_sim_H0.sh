#!/bin/bash
#SBATCH --job-name=unity
#SBATCH --partition=shared
#SBATCH --time=1-06:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
cd /home/drubin/union3/simulated

python step1_make_simLCs.py --ndataset 20 --addnoise 1 --addcalibration 1 --modeluncertainty 1 --prefixname /home/drubin/koa_scratch/sim_data_simplex_H0 --skewdist 1 --volumelimited 0 --obsmagselection 1 --zrangekeys SLHV --sigzp 0.01 --ncalibperset 20 --sigmabetaR 0.0 --nnearbyperset 200 --nvisit 100

cd /home/drubin/koa_scratch/sim_data_simplex_H0
python /home/drubin/salt2_union//python_code/slurmfit.py 10 dontsort
