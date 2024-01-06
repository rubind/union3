try:
    from commands import getoutput
except:
    from subprocess import getoutput
import sys
import glob

def check_param(logfl, param, leave_running_jobs_alone):
    grepout = getoutput("grep '" + param + " ' " + logfl).split(None)
    print(grepout)
    
    if grepout == []:
        return leave_running_jobs_alone # Still running and leave_running_jobs_alone => 1
    elif float(grepout[-1]) < 1.05:
        return 1
    else:
        return 0
    

leave_running_jobs_alone = int(sys.argv[1])

for logfl in glob.glob("UNITY*/log.txt"):
    check_Om = check_param(logfl, param = "Om", leave_running_jobs_alone = leave_running_jobs_alone)
    check_wDE = check_param(logfl, param = "wDE", leave_running_jobs_alone = leave_running_jobs_alone)

    if check_Om and check_wDE:
        print("Good!")
    else:
        cmd = "cd " + logfl.split("/")[0] + "\nsbatch run.sh"
        print(cmd)
        print(getoutput(cmd))
