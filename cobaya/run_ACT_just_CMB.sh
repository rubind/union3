#!/bin/bash
#SBATCH --job-name=cobaya
#SBATCH --partition=shared
#SBATCH --time=2-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
source activate cobaya_w0wa

cd /home/drubin/union3/cobaya
cobaya-run ACT_just_CMB.input.yaml -p ~/cobaya_packages -o chain_ACT_just_CMB --resume


