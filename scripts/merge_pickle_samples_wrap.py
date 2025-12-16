from subprocess import getoutput
import os
import glob

pwd = getoutput("pwd")



unique_prefixes = {}

for logfl in getoutput("ls -lthr *cosmo=*/log.txt | grep 'M '").split('\n'):
    logfl = logfl.split(None)[-1]
    print(logfl)

    prefix = logfl.split("/")[0][:-3]
    if prefix in unique_prefixes:
        unique_prefixes[prefix].append(logfl)
    else:
        unique_prefixes[prefix] = [logfl]

print("unique_prefixes", unique_prefixes)

for prefix in unique_prefixes:
    print("Checking sample size")
    sample_file = glob.glob(unique_prefixes[prefix][0].split("/")[0] + "/sample*pickle")
    assert len(sample_file) == 1
    sample_file = sample_file[0]

    print("sample_file", sample_file)
    the_size = os.path.getsize(sample_file)
    print(sample_file, the_size)
    n_files = len(unique_prefixes[prefix])

    print("n_files", n_files)

    if the_size > 100e6:
        # Looks like all parameters saved. Must be for PPD. But also want unthinned for uncertainty analysis.
        # How much to thin? for 100 runs, 47. For 20 runs, 11.

        if n_files > 50:
            thins = [0, 47]
        else:
            thins = [0, 11]
            
        max_params = [1000, 10000]
    else:
        thins = [0]
        if prefix.count("cosmo=2"):
            max_params = [100] # Only want cosmology samples
        else:
            max_params = [1000] # Want everything but per-SN parameters
        
    for max_param, thin in zip(max_params, thins):
        f = open("merge.sh", 'w')
        f.write("""#!/bin/bash
#SBATCH --job-name=merge                                      
#SBATCH --partition=shared
#SBATCH --time=00-14:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile

module load lang/Anaconda3/2023.03-1
conda info --envs
conda activate py39
""")
        
        f.write("cd " + pwd + '\n')
        the_files = [item.split("/")[0] + "/" + sample_file.split("/")[-1] for item in unique_prefixes[prefix]]
        the_files.sort()
        
        f.write("python $UNITY/scripts/merge_pickle_samples.py %i %i %s\n" % (max_param, thin, " ".join(the_files)))
        f.close()
    
        print(getoutput("sbatch merge.sh"))
    
    
