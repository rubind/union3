from commands import getoutput
import time
#import tqdm

for i in range(100):
    f = open("tmp.sh", 'w')
    f.write("""#!/bin/bash
#SBATCH --job-name=unity
#SBATCH --partition=shared
#SBATCH --time=00-03:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
cd /home/drubin/union3/simulated/2d_sim_test/1D_int
python 2d_sim.py > log""" + str(i) + ".txt")
    f.close()
    time.sleep(0.25)

    getoutput("sbatch tmp.sh")
