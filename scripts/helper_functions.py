from numpy import *
from scipy.stats import scoreatpercentile
from scipy.interpolate import interp1d
import os
import commands
from string import strip

################################################# File-Reading Functions ###################################################
def clean_lines(lines, stringlist = [""]):
    # Start by getting rid of spaces.

    lines = [item.strip() for item in lines]
    
    # Check for strings to exclude.
    lines = [item for item in lines if stringlist.count(item) == 0]

    # Get rid of comments
    lines = [item for item in lines if item[0] != "#"]
    return lines

    

def read_param(flnm, param, default = None, ind = 1):
    # This one is commonly used for photometry parameter files.
    
    fp = open(flnm)
    lines = fp.read()
    fp.close()

    lines = lines.split('\n')
    lines = clean_lines(lines)

    for line in lines:
        parsed = line.split(None)
        if parsed[0] == param:
            print "Reading " + param + " from " + flnm

            try:
                # Yeah, I know eval is bad. But it works with all types!
                return eval(parsed[ind])
            except:
                return parsed[ind]
    print 
    print "Couldn't find ", param, flnm
    print "Returning default ", default
    print

    return default


def get_params(paramfl):
    print "Reading params from ", paramfl

    f = open(paramfl)
    lines = f.read().split('\n')
    f.close()

    keys = ["filenamelist", "weird_sn_list", "mag_cut", "calib_errs",
            "iter", "chains", "n_jobs",
            "max_firstphase", "min_lastphase", "max_color_uncertainty",
            "min_redshift", "max_redshift",
            "do_blind", "do_twoalphabeta", "outl_frac",
            "stan_code", "pec_vel_disp", "lensing_disp", "sample_file", "log10_sigma_int_mB"
            ]
    params = {}

    for line in lines:
        parsed = line.split(None)
        if len(parsed) > 1:
            for key in keys:
                if parsed[0] == key:
                    if len(parsed) == 2:
                        try:
                            params[key] = eval(parsed[1])
                        except:
                            params[key] = parsed[1]
                    else:
                        params[key] = parsed[1:]

    for key in keys:
        if not params.has_key(key):
            print "Didn't read ", key, "setting to None"
            params[key] = None

    if params["sample_file"] != None:
        params["sample_file"] = os.getcwd() + "/" + params["sample_file"]
    for key in ["weird_sn_list", "mag_cut", "stan_code"]:
        if params[key].count("$"):
            print params[key]
            params[key] = commands.getoutput("ls " + params[key])
            print "->", params[key]
    for i in range(len(params["filenamelist"])):
        if params["filenamelist"][i].count("$"):
            print params["filenamelist"][i]
            params["filenamelist"][i] = commands.getoutput("ls " + params["filenamelist"][i])
            print "->", params["filenamelist"][i]


    print "Read params ", params
    assert isinstance(params["filenamelist"], list), "filenamelist should be a list!"
    return params



def get_dparam_dzps(res_der_fl):
    f = open(res_der_fl)
    lines = f.read().split('\n')
    f.close()

    dparam_dzps = {}
    for line in lines:
        parsed = line.split(None)
        if parsed.count("All") and (parsed.count("Zeropoint") or parsed.count("Lambda")):
            parsed = line.split(None)
            assert parsed[3] == "All", "Weird format for " + res_der_fl
            
            dparam_dzps[(parsed[0], parsed[1][parsed[1].find("|")+1:])] = array([float(parsed[5]), float(parsed[6]), float(parsed[7])])

    print "dparam_dzps ", dparam_dzps
    return dparam_dzps


def get_calib_uncertainties(calib_names, zeropointfl):
    assert 0, "Deprecated!"

    calib_uncertainties = []

    f = open(zeropointfl)
    lines = f.read().split('\n')
    f.close()

    print lines
    lkfjds

    
    for calib_name in calib_names:
        found = 0

        for line in lines:
            parsed = line.split(None)

            if parsed.count(calib_name[1]):
                if calib_name[0] == "Zeropoint":
                    calib_uncertainty = float(parsed[1])
                elif calib_name[0] == "Lambda":
                    calib_uncertainty = float(parsed[2])
                print "Applying ", calib_uncertainty, "for", calib_name
                calib_uncertainties.append(calib_uncertainty)
                found += 1
        assert found == 1, "Found zero times or more than once! " + str(calib_name) + " " + str(found)

    return calib_uncertainties

def samples_txt_to_pickle(flname, samples_to_burn, skip = 6):
    """NOTE: these samples come out in a different order than the extracted fit!"""

    f = open(flname)
    lines = f.read().split('\n')
    f.close()

    headings_size = {}
    headings = []

    for i in range(6):
        parsed = lines[i].split(",")[skip:]
        if len(parsed) > 1 and len(headings_size.keys()) == 0:
            headings = parsed
            for j in range(len(parsed)):
                pparsed = parsed[j].split(".")
                new_key = not headings_size.has_key(pparsed[0])

                if parsed[j].count(".") == 0:
                    headings_size[pparsed[0]] = 0
                elif parsed[j].count(".") == 1:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]),)
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     )
                elif parsed[j].count(".") == 2:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]), int(pparsed[2]))
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     max(headings_size[pparsed[0]][1], int(pparsed[2]))
                                                     )
                elif parsed[j].count(".") == 3:
                    if new_key:
                        headings_size[pparsed[0]] = (int(pparsed[1]), int(pparsed[2]), int(pparsed[3]))
                    else:
                        headings_size[pparsed[0]] = (max(headings_size[pparsed[0]][0], int(pparsed[1])),
                                                     max(headings_size[pparsed[0]][1], int(pparsed[2])),
                                                     max(headings_size[pparsed[0]][2], int(pparsed[3]))
                                                     )
                else:
                        assert 0, "!!!!!!" + parsed[j]

    samples_count = -samples_to_burn
    
    for line in lines:
        parsed = line.split(",")[skip:]
        if len(parsed) == len(headings):
            try:
                float(parsed[0])
                samples_count += 1
            except:
                pass

    print "samples_count ", samples_count

    print headings_size
    samples = {}

    for key in headings_size:
        if headings_size[key] == 0:
            samples[key] = zeros(samples_count, dtype=float64) - 1.e20
        else:
            samples[key] = zeros((samples_count,) + headings_size[key], dtype=float64)  - 1.e20
                    
    linecount = -samples_to_burn - 1
    for line in lines:
        parsed = line.split(",")[skip:]

        if len(parsed) == len(headings):
            try:
                float(parsed[0])
                skip_line = 0
            except:
                skip_line = 1

            if not skip_line:
                linecount += 1
                if linecount >= 0:
                    for j in range(len(parsed)):
                        pparsed = headings[j].split(".")
                        
                        if len(pparsed) == 1:
                            samples[pparsed[0]][linecount] = float(parsed[j])
                        else:
                            key = tuple([linecount] + [int(item) - 1 for item in pparsed[1:]])
                            samples[pparsed[0]][key] = float(parsed[j])
    return samples


def get_kcorrect_ifns(magcut_k_correction_fl):
    from FileRead import readcol

    [z, c2, c3] = readcol(magcut_k_correction_fl, 'fff')
    return interp1d(z, c2, kind = 'linear'), interp1d(z, c3, kind = 'linear')


################################################# Chain Functions ###################################################

def gelman_rubin_R(samples):
    """samples should be an array (nsamples = n, nchains = m). This is the original formula without the sqrt!"""
    n = len(samples)
    print "nsamples ", n
    m = len(samples[0])
    print "nchains ", m

    psi_dotj = mean(samples, axis = 0)
    psi_dotdot = mean(psi_dotj)
    s_j2 = 1./(n - 1.) * sum((samples - psi_dotj)**2.)

    B = n/(m - 1.) * sum((psi_dotj - psi_dotdot)**2.)
    W = (1./m)*sum(s_j2)

    var_t = (n - 1.)*W/n + B/n
    return var_t/W



def filter_fit_params(fit_params, param_name, chains, iter_per_chain):

    stdevs = array([], dtype=float64)
    for i in range(chains):
        stdevs = append(stdevs, std(fit_params[param_name][i*iter_per_chain:(i+1)*iter_per_chain])
                        )
    print param_name, "stdevs ", stdevs
    good_chains = stdevs > max(stdevs)/4.
    print "good_chains", good_chains
    
    good_inds = []
    for i in range(chains):
        if good_chains[i]:
            good_inds.extend(range(i*iter_per_chain, (i+1)*iter_per_chain))

    for key in fit_params:
        print key, fit_params[key].shape,
        try:
            fit_params[key] = fit_params[key][good_inds]
        except:
            print "Error!"
            sys.exit(1)
        print fit_params[key].shape
    return fit_params


def quick_print(vals, thename):
    txt = "  ".join(["SummaryPrint ", thename, "Mean 15.9 50 84.1", "%.4f" % mean(vals), "%.4f" % scoreatpercentile(vals, 15.8655), "%.4f" % scoreatpercentile(vals, 50), "%.4f" % scoreatpercentile(vals, 84.1345)])
    return txt


def summarize_parameters(fit_params, thekeys = None, on_screen = 1):
    if thekeys == None:
        thekeys = fit_params.keys()

    txt = ""

    for parameter in thekeys:
        the_shape = fit_params[parameter].shape

        if len(the_shape) == 1:
            txt += quick_print(fit_params[parameter], parameter) + '\n'
        elif len(the_shape) == 2:
            for i in range(the_shape[1]):
                txt += quick_print(fit_params[parameter][:,i], parameter + "::" + str(i)) + '\n'
        elif len(the_shape) == 3:
            for i in range(the_shape[1]):
                for j in range(the_shape[2]):
                    txt += quick_print(fit_params[parameter][:,i,j], parameter + "::" + str(i) + "::" + str(j)) + '\n'
        else:
            pass
    
    if on_screen:
        print txt
    return txt

