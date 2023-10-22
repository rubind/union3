import subprocess
import time

for cosmo in """6 1 flatLCDM
6 1 flatwCDM
6 1 LCDM
6 1 flatw0wa
6 1 w0wa
6 1 flatw0waEDE
6 1 w0waEDE
4 1 flatw0waOmh
4 1 flatw0waOmhEDE""".split('\n'):

    f = open("tmp.sh", 'w')
    f.write("""#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=02-00:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

module load lang/Anaconda3/2023.03-1

conda info --envs
conda activate py39

cd /home/drubin/union3/other_cosmology
~/.conda/envs/py39/bin/python3.9 compute_chi2s.py """ + cosmo)
    f.close()

    print(subprocess.getoutput("sbatch tmp.sh"))
    
    time.sleep(1)
