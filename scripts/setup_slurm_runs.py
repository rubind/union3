import sys
from subprocess import getoutput
import glob
import gzip
import pickle


def do_it(cmd):
    print(cmd)
    print(getoutput(cmd))
    

def check_if_good(wd, par_name):
    if glob.glob(wd + "/log.txt") == []:
        print("Couldn't check", wd)
        return 0
    
    grepout = getoutput("grep %s %s/log.txt" % (par_name, wd)).split('\n')

    did_check = leave_running_jobs_alone
    
    for line in grepout:
        parsed = line.split(None)
        
        if len(parsed) > 4 and parsed[0].count(par_name) == 1:
            did_check = 1
            
            neff = float(parsed[-2])

            print(parsed, neff)

            if neff < 100:
                print("Bad!", wd)
                return 0
    return did_check
    
pwd = getoutput("pwd")
whoami = getoutput("whoami")

assert getoutput('squeue | grep drubin | grep -v " R "').strip() == ""

print("python union3/scripts/setup_slurm_runs.py union3/scripts/union3 koa_scratch/Union3 100 1356 0 1")

orig_dir = sys.argv[1]
new_dir_loc = sys.argv[2]
n_copy = int(sys.argv[3])
cosmomodels = [int(item) for item in sys.argv[4]]
redo_runs = int(sys.argv[5])
leave_running_jobs_alone = int(sys.argv[6])



for i in range(n_copy):
    suffix = orig_dir.split("/")[-1]
    assert len(suffix) > 3

    for cosmomodel in cosmomodels:#[6]:#[1,6,3,5]:
        wd = new_dir_loc + "/" + suffix + "_cosmo=%i_%03i" % (cosmomodel, i)

        good_run = 1

        good_run *= check_if_good(wd, "Om")
        good_run *= check_if_good(wd, "mu_zbins")

        errglob = glob.glob(wd + "/*.err")
        if len(errglob) > 0:
            f = open(errglob[0], 'r')
            lines = f.read()
            f.close()

            print(lines)
            
            if lines.count("DUE TO TIME LIMIT"):
                good_run = 0
                print("TIME LIMIT")

        
        if redo_runs or (good_run == 0):
            do_it("rm -fr " + wd)
            do_it("mkdir -p " + wd)
            do_it("cp " + orig_dir + "/inputs*pickle " + wd)
            do_it("cp " + orig_dir + "/*stan*txt " + wd)

            
            pfl = glob.glob(wd + "/inputs*pickle")
            assert len(pfl) == 1
            pfl = pfl[0]

            (the_data, stan_data, params) = pickle.load(gzip.open(pfl, 'rb'))

            if cosmomodel != 1:
                print("cosmomodel", cosmomodel, " so restricting saved parameters!")
                params["max_params_to_save"] = min(params["max_params_to_save"], 1000)

                pickle.dump((the_data, stan_data, params), gzip.open(pfl, "wb"))


            if cosmomodel != 6:
                time_limit = "03-00:00:00"
            else:
                time_limit = "03-00:00:00"

                        
            f = open(wd + "/run.sh", 'w')
            f.write("""#!/bin/bash
#SBATCH --job-name=example
#SBATCH --partition=shared
#SBATCH --time=""" + time_limit + """ ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
export REALLYUNBLIND=1                                                                                                           
module load lang/Anaconda3/2023.03-1

conda info --envs
conda activate py39

cd """ + wd + """
~/.conda/envs/py39/bin/python3.9 $UNITY/scripts/read_and_sample.py inputs*pickle %i > log.txt
""" % cosmomodel)
            f.close()
            do_it("cd " + wd + '\n' + "sbatch run.sh")
