try:
    from commands import getoutput
except:
    from subprocess import getoutput
import sys
import glob

def check_param(logfl, param, leave_running_jobs_alone, threshold = 1.05):
    grepout = getoutput("grep '" + param + " ' " + logfl).split(None)
    print(grepout)
    
    if grepout == []:
        return leave_running_jobs_alone # Still running and leave_running_jobs_alone => 1
    elif float(grepout[-1]) < threshold:
        return 1
    else:
        return 0
    

leave_running_jobs_alone = int(sys.argv[1])

whoami = getoutput("whoami")
grepout = getoutput("squeue | grep " + whoami + " | grep -v ' R '")

assert grepout.strip() == "", "Some jobs are still queued"

for logfl in glob.glob("UNITY*/log.txt"):
    check_Om = check_param(logfl, param = "Om", leave_running_jobs_alone = leave_running_jobs_alone)
    check_wDE = check_param(logfl, param = "wDE", leave_running_jobs_alone = leave_running_jobs_alone)

    other_checks = 1
    for key in ["beta_B", "mBx1c_int_variance\[1\]"]:
        other_checks *= check_param(logfl, param = key, leave_running_jobs_alone = leave_running_jobs_alone, threshold = 1.2)

        
    if check_Om and check_wDE and other_checks:
        print("Good!")
    else:
        cmd = "cd " + logfl.split("/")[0] + "\nsbatch run.sh"
        print(cmd)
        print(getoutput(cmd))
