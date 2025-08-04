try:
    from commands import getoutput
except:
    from subprocess import getoutput
import sys
import glob

def check_param(logfl, param, leave_running_jobs_alone, threshold = 1.05):
    grepout = getoutput("grep -a '" + param + " ' " + logfl).split(None)
    print(grepout)
    
    if grepout == []:
        return leave_running_jobs_alone # Still running and leave_running_jobs_alone => 1
    elif float(grepout[-1]) < threshold:
        return 1
    else:
        return 0

def check_calibs(logfl, leave_running_jobs_alone, threshold):
    grepout = getoutput("grep -a 'calibs\[' " + logfl).split('\n')
    print(grepout)

    if len(grepout) < 2:
        return leave_running_jobs_alone # Still running and leave_running_jobs_alone => 1

    for line in grepout:
        parsed = line.split(None)
        if float(parsed[-1]) > threshold:
            return 0
    return 1


def check_sampling(logfl):
    # distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1
    errfls = glob.glob(logfl.replace("log.txt", "*.err"))

    
    for errfl in errfls:
        print("Checking", errfl)
        grepout = getoutput("grep -a " + '"' + "distutils.errors.DistutilsExecError: command '/usr/bin/gcc' failed with exit code 1" + '" ' + errfl)
        if grepout.count("DistutilsExecError") == 1:
            return 0

        grepout = getoutput("grep -a 'CANCELLED AT' " + errfl)
        if grepout.count("CANCELLED AT") == 1:
            return 0
        
    return 1


leave_running_jobs_alone = int(sys.argv[1])

whoami = getoutput("whoami")
grepout = getoutput("squeue | grep " + whoami + " | grep -v ' R ' | grep -v ' CG ' ")

assert grepout.strip() == "", "Some jobs are still queued"

if len(sys.argv) == 2:
    logfls = glob.glob("UNITY*/log.txt")
else:
    logfls = sys.argv[2:]

for logfl in logfls:
    check_Om = check_param(logfl, param = "Om", leave_running_jobs_alone = leave_running_jobs_alone)
    check_wDE = check_param(logfl, param = "wDE", leave_running_jobs_alone = leave_running_jobs_alone)

    other_checks = 1
    for key in ["beta_B", "mBx1c_int_variance\[1\]", "beta_R_low", "beta_R_high", "alpha_fast", "alpha_slow", "alpha", "MB_fast_minus_slow"]:
        other_checks *= check_param(logfl, param = key, leave_running_jobs_alone = leave_running_jobs_alone, threshold = 1.2)

    other_checks *= check_calibs(logfl, leave_running_jobs_alone = leave_running_jobs_alone, threshold = 1.2)
    other_checks *= check_sampling(logfl)
    
        
    if check_Om and check_wDE and other_checks:
        print("Good!")
    else:
        wd = "/".join(logfl.split("/")[:-1])
        getoutput("rm -f " + wd + "/samples*pickle")
        getoutput("rm -f " + wd + "/log.txt")
        getoutput("rm -f " + wd + "/*.err")
        getoutput("rm -f " + wd + "/*.out")

        cmd = "cd " + wd + "\nsbatch run.sh"
        print(cmd)
        print(getoutput(cmd))

print("Done checking!")
