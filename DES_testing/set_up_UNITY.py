import sys
import subprocess

def do_it(cmd):
    print(cmd)
    print(subprocess.getoutput(cmd))
    
pwd = subprocess.getoutput("pwd")

for dr in sys.argv[1:]:
    print(dr)

    wd = dr + "/UNITY"
    do_it("rm -fr " + wd)
    do_it("mkdir " + wd)

    f = open(wd + "/paramfile.txt", 'w')
    f.write("""do_blind		0

filenamelist		["$UNION/DES_sim_Shallow_v1.txt","$UNION/DES_sim_Deep_v1.txt"]


weird_sn_list		"$UNITY/paramfiles/weird_sn_list.txt"
mag_cut			"$UNITY/paramfiles/mag_cuts.txt"
stan_code		"$UNITY/scripts/stan_code_H0.txt"
sample_file		"None"
calibration_uncertainties    "calibration_uncertainties.txt"

max_params_to_save	     20

min_redshift		0.01
max_redshift		1.2
max_firstphase		100.
min_lastphase		-100.
max_color_uncertainty	0.2
max_color		0.3
max_MWEBV		0.3
min_color		-0.3
remap_x1		[0.,0.]
distance_ladder		None

# Units of c:
pec_vel_disp		0.001
# Units of magnitudes per redshift
lensing_disp		0.055
MWEBV_zeropoint_EBV	0.005
outl_frac		0.02


redshift_coeff_type     sample 0.4

electron_coeff		[0.000042,0.0000042]
IG_extinction_coeff	0.001


do_twoalphabeta		1
threeD_unexplained	1

iter			2500
n_jobs			4
chains			4

do_host_mass		1
fix_Om			0
MB_by_sample		0
include_pec_cov		0
separate_mass_x1c	1
""")
    f.close()

    
    f = open(wd + "/calibration_uncertainties.txt", 'w')
    f.write("""
('Fundamental', (3000.0, 4000.0)):                                                                      0.0001
('Fundamental', (4000.0, 5000.0)):                                                                      0.0001
('Fundamental', (6000.0, 8000.0)):                                                                      0.0001
('Fundamental', (8000.0, 100000.0)):                                                                    0.0001
('Fundamental', (10000.0, 100000.0)):                                                                   0.0001


"SALT_UV_CAL":                                                                                          0.0001
"SALT_U_CAL":                                                                                           0.0001
"SALT_I_CAL":                                                                                           0.0001

('Zeropoint', 'DECam|DECam_g'): 0.0001
('Zeropoint', 'DECam|DECam_r'): 0.0001
('Zeropoint', 'DECam|DECam_i'): 0.0001
('Zeropoint', 'DECam|DECam_z'): 0.0001
('Lambda', 'DECam|DECam_g'):    0.0001
('Lambda', 'DECam|DECam_r'):    0.0001
('Lambda', 'DECam|DECam_i'):    0.0001
('Lambda', 'DECam|DECam_z'):    0.0001
""")
    
    f.close()



    f = open(wd + "/run.sh", 'w')
    f.write("""#!/bin/bash
#SBATCH --job-name=UNITY
#SBATCH --partition=shared
#SBATCH --time=1-18:00:00 ## time format is DD-HH:MM:SS
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G # Memory per node my job requires
#SBATCH --error=example-%A.err # %A - filled with jobid, where to write the stderr
#SBATCH --output=example-%A.out # %A - filled with jobid, wher to write the stdout
source ~/.bash_profile
export UNION=../
""")
    f.write("cd " + pwd + '\n')
    f.write("cd " + wd + '\n')
    f.write("~/.conda/envs/py39/bin/python $UNITY/scripts/read_and_sample_H0.py paramfile.txt 1 > log.txt\n")
    f.close()

    
    do_it("cd " + wd + "\nsbatch run.sh")
    
