import numpy as np
from scipy.interpolate import interp1d
from astropy.io import ascii
from FileRead import readcol
import os

# Ports/copies of code written for minimarginalize.py for Union compilations

def get_colors(key):
    if key == "blue":
        return ((0., 83/255., 152/255.),
                (30/255., 104/255., 168/255.),
                (92/255., 140/255., 190/255.))

    if key == "orange":
        return ((243/255., 116/255., 17/255.),
                (248/255., 144/255., 62/255.),
                (250/255., 180/255., 110/255.))

    if key == "green":
        return ((0., 160/255., 52/255.),
                (50/255., 176/255., 86/255.),
                (117/255., 198/255., 126/255.))
              
    if key == "gray":
        return ((71/255., 71/255., 71/255.),
                (119/255., 119/255., 119/255.),
                (178/255., 178/255., 178/255.))

    if key == "teal":
        return ((0/255., 168/255., 168/255.),
                (127/255., 204/255., 189/255.),
                (190/255., 226/255., 210/255.))
    assert 0, key



################################## CMB/ BAO Functions #################################

def get_sound_speed(z, O_bhh):
    return 1./np.sqrt(3.*(1. + 0.75*O_bhh/CosConst.O_gammahh/(1. + z)))


def getzstar(O_m, hh, O_bhh):
    g1 = (0.0783*(O_bhh)**(-0.238))/(1. + 39.5*O_bhh**0.763)
    g2 = 0.560/(1. + 21.1*O_bhh**1.81)
    
    #ztar is CMB redshift
    zstar = 1048.*(1. + 0.00124*O_bhh**(-0.738))*(1 + g1*(O_m*hh)**g2)

    return zstar


def getzdrag(O_m, hh, O_bhh):
    """zdrag is BAO redshift"""
    
    b1 = 0.313*(O_m*hh)**(-0.419)*(1. + 0.607*(O_m*hh)**(0.674))
    b2 = 0.238*(O_m*hh)**0.223
    
    zdrag = 1291*(O_m*hh)**(0.251)*(1. + b1*(O_bhh)**b2)/(1. + 0.659*(O_m*hh)**0.828)

    return zdrag


def highz_r(cosmo, zmin, zmax, sound = 0): # use log(1 + z) instead of z to to the integral-- faster for large z

    # for matter only, 10**8 gives 0.996836 of real result
    # for matter only, 10**9 gives 0.999 of real result

    # matter + rad, 10**5 gives 0.985469 of real result
    # matter + rad, 10**6 gives 0.998538 of real result
    # matter + rad, 10**7 gives 0.999854 of real result

    
    n = 300
    zint_list = np.exp(np.linspace(np.log(1 + zmin), np.log(1 + zmax), n)) - 1
    zint_list_inter = 0.5*(zint_list[1:] + zint_list[:-1])


    Hinvlist = get_Hinv(zint_list, cosmo)
    Hinvlist_inter = get_Hinv(zint_list_inter, cosmo)


    if sound:
        Hinvlist *= get_sound_speed(zint_list, cosmo["O_bhh"])
        Hinvlist_inter *= get_sound_speed(zint_list_inter, cosmo["O_bhh"])


    r = 0.
    for i in range(n - 1):
        r += (zint_list[i + 1] - zint_list[i])*(Hinvlist[i] + 4.*Hinvlist_inter[i] + Hinvlist[i + 1])/6.

    omega_k = cosmo["O_k"]
    r = sinn_r(r, cosmo["O_k"])

    return r

def get_sound_horizon(cosmo, CMB_not_BAO):
    """r* for CMB, r_drag for BAO"""

    if CMB_not_BAO:
        zmin = getzstar(cosmo["O_m"], cosmo["h"]**2, cosmo["O_bhh"])
    else:
        zmin = getzdrag(cosmo["O_m"], cosmo["h"]**2, cosmo["O_bhh"])
        
    return highz_r(cosmo = cosmo, zmin = zmin, zmax = 10**9, sound = 1)


def get_R(cosmo):
    r = highz_r(cosmo, zmin = 0, zmax = getzstar(cosmo["O_m"], cosmo["h"]**2, cosmo["O_bhh"]), sound = 0)   
    
    return np.sqrt(cosmo["O_m"]) * r


def Planck18_CMB_chi2(cosmo, merged_mat):
    R = get_R(cosmo)

    z_star = getzstar(O_m = cosmo["O_m"], hh = cosmo["h"]**2., O_bhh = cosmo["O_bhh"])
    r_star = get_sound_horizon(cosmo = cosmo, CMB_not_BAO = 1)
    theta = 100 * r_star / highz_r(cosmo = cosmo, zmin = 0., zmax = z_star)

    
    vals_0 = merged_mat[-1]
    resid = np.array([R, theta, cosmo["O_bhh"]]) - vals_0

    return np.dot(resid, np.dot(merged_mat[3:6], resid))

def load_BAO():
    [constraint, redshift, fiducial, value] = readcol(os.environ["UNITY"] + "/other_cosmology/BAO_results.txt", 'afff')

    f = open(os.environ["UNITY"] + "/other_cosmology/BAO_results.txt", 'r')
    lines = f.read().split('\n')
    f.close()

    lines = [line for line in lines if len(line.split(None)) > 1]
    
    cov_mat = []
    for line in lines[1:]:
        cov_mat.append([float(item) for item in line.split(None)[6:]])
    cov_mat = np.array(cov_mat)

    print("cov_mat", cov_mat)
    assert cov_mat.shape[0] == cov_mat.shape[1]
    assert (np.isclose(cov_mat - cov_mat.T, 0)).all()
    w_mat = np.linalg.inv(cov_mat)

    BAO_data = dict(constraint = constraint, redshift = redshift, fiducial = fiducial, value = value, w_mat = w_mat)
    return BAO_data


def get_DV(z, cosmo):
    r = n_integrate(z, get_Hinv, cosmo)[0]
    r = sinn_r(r, cosmo["O_k"])

    DV = (r**2. * z*get_Hinv(z, cosmo) ) ** (1./3.)
    return DV
    
def get_DM(z, cosmo):
    r = n_integrate(z, get_Hinv, cosmo)[0]
    r = sinn_r(r, cosmo["O_k"])

    return r

def get_H(z, cosmo):
    return 1./get_Hinv(z, cosmo)



def get_BAO_chi2(BAO_data, cosmo, verbose = False):
    rs = get_sound_horizon(cosmo, CMB_not_BAO = 0)

    resid = []
    for i in range(len(BAO_data["constraint"])):
        if BAO_data["constraint"][i] == "DV":
            this_val = get_DV(BAO_data["redshift"][i], cosmo)/rs
            this_val *= BAO_data["fiducial"][i]
        elif BAO_data["constraint"][i] == "DM":
            this_val = get_DM(BAO_data["redshift"][i], cosmo)/rs
            this_val *= BAO_data["fiducial"][i]
        elif BAO_data["constraint"][i] == "H":
            this_val = get_H(BAO_data["redshift"][i], cosmo)*rs*CosConst.c100_Mpc * 100. # h cancels out
            this_val /= BAO_data["fiducial"][i]
        else:
            assert 0

        if verbose:
            print("this_val", this_val, BAO_data["value"][i])
        resid.append(this_val - BAO_data["value"][i])
    resid = np.array(resid)
    chi2 = np.dot(resid, np.dot(BAO_data["w_mat"], resid))

    #c_mat = np.linalg.inv(BAO_data["w_mat"])
    #uncs = np.sqrt(np.diag(c_mat))
    #print(resid/uncs)

        
    return chi2

################################## End CMB/ BAO Functions #################################



def n_integrate(x_list, intfn, otherargs, pad_list = np.arange(41, dtype=np.float64)/20.):

    nadded = len(pad_list)
    x_list = np.append(x_list, pad_list) # make sure that these values are evaluated

    if len(x_list) % 2 == 0:
        nadded += 1
        x_list = np.append(x_list, 2.1) # make sure it's odd

    x_indices = np.argsort(x_list)
    x_sort_list = np.take(x_list, x_indices)

    #print x_sort_list

    #2n + 1 samples

    #(1/3)*(dx)*(f0 + 4*(f1 + f3 + ... + f2n-1) + 2*(f2 + f4 + ... + f2n-2) + f2n)


    intfn_sort = intfn(x_sort_list, otherargs) # (2n + 1)

    interxlist = 0.5*(x_sort_list[:-1] + x_sort_list[1:])
    
    intfn_sort_inter = intfn(interxlist, otherargs)  # (2n)


    the_integral = np.zeros(len(intfn_sort), dtype=np.float64)

    
    for i in range(1,len(intfn_sort)):
        the_integral[i] = the_integral[i - 1] + (intfn_sort[i - 1] + 4.*intfn_sort_inter[i - 1] + intfn_sort[i])*(x_sort_list[i] - x_sort_list[i - 1])/6.


    tmplist = the_integral*1.
    np.put(the_integral, x_indices, tmplist)

    the_integral = the_integral[:-nadded]

    return the_integral


def get_Hinv(z_list, cosmo):
    omega_r = CosConst.O_radhh/cosmo["h"]**2.

    if cosmo["model"] == "LCDM":
        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + cosmo["O_m"]*(1. + z_list)**3. + (1. - cosmo["O_m"] - cosmo["O_k"] - omega_r) + cosmo["O_k"]*(1. + z_list)**2.
            )

    if cosmo["model"] == "flatLCDM":
        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + cosmo["O_m"]*(1. + z_list)**3. + (1. - cosmo["O_m"] - omega_r)
            )

    if cosmo["model"] == "flatwCDM":
        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + cosmo["O_m"]*(1. + z_list)**3. + (1. - cosmo["O_m"] - omega_r)*(1 + z_list)**(3.*(1. + cosmo["w"]))
            )
    
    elif cosmo["model"] == "flatw0wa" or cosmo["model"] == "w0wa":
        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + cosmo["O_m"]*(1. + z_list)**3. + cosmo["O_k"]*(1. + z_list)**2.
            + (1. - cosmo["O_m"] - cosmo["O_k"] - omega_r)*np.exp(-3.0*cosmo["w_a"]*z_list/(1. + z_list))*(1. + z_list)**(3.0*(1.0 + cosmo["w_0"] + cosmo["w_a"]))
            )

    elif cosmo["model"] == "binnedrho":
        DE_density = (z_list <= cosmo["zbins"][0])*(1. - cosmo["O_m"] - cosmo["O_k"] - omega_r)
        DE_density += (z_list > cosmo["zbins"][-1])*cosmo["rhobins"][-1]
        
        for i in range(len(cosmo["zbins"]) - 1):
            DE_density += (z_list > cosmo["zbins"][i])*(z_list <= cosmo["zbins"][i+1])*cosmo["rhobins"][i]

        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + cosmo["O_m"]*(1. + z_list)**3. + cosmo["O_k"]*(1. + z_list)**2. + DE_density)

    
    elif len(cosmo) == 5:
        w_p = cosmo[2]
        w_a = cosmo[3]
        z_p = cosmo[4]
        
        return 1./np.sqrt(
            omega_r*(1. + z_list)**4. + O_M*(1. + z_list)**3. + O_k*(1. + z_list)**2. + O_x*exp(-3.0*w_a*z_list/(1. + z_list))*(1. + z_list)**(3.0*(1.0 + w_p + w_a/(1. + z_p)))
            )
    else:
        assert 0, "Unknown model!"



def get_dt(z_list, cosmo):
    return get_Hinv(z_list, cosmo)/(1. + z_list)


def get_t_look(z_list, cosmo):
    return n_integrate(z_list, get_dt, cosmo)


def sinn_r(r_list, O_k):
    if abs(O_k) < 1.e-6:
        return r_list
    
    if O_k > 0.:
        return np.sinh(np.sqrt(np.abs(O_k))  *  r_list)  /  np.sqrt(np.abs(O_k))
    else:
        return np.sin(np.sqrt(np.abs(O_k))  *  r_list)  /  np.sqrt(np.abs(O_k))

def get_mu(z_list, cosmo, z_helio_list = [-2]):
    r_list = n_integrate(z_list, get_Hinv, cosmo)

    if z_helio_list[0] < -1:
        z_helio_list = z_list
    
    return 5.*np.log10(  (1. + z_helio_list)*sinn_r(r_list, cosmo["O_k"])  ) + CosConst.cH100 - 5*np.log10(cosmo["h"])


"""
def mu_to_z(mu_list, cosmo): # I'll have to generalize this later
    cH0 = CosConst.cH70

    new_z = 10.**(0.2*(mu_list - cH0))
    old_z  = 1000.

    while max(abs(new_z - old_z)) > 1.e-3:
        mu_new = get_mu(new_z, cosmo) - mu_list
        deriv_new = get_mu(new_z + 0.01, cosmo) - mu_list
        deriv_new = (deriv_new - mu_new)/0.01

        old_z = copy.deepcopy(new_z)
        new_z += - mu_new/deriv_new

        if min(new_z) < 0 or max(new_z) > 100.:
            print "Error! ", mu_list, cosmo
            sys.exit(1)

    return new_z
"""
def mu_to_z(mu_list, cosmo): # I'll have to generalize this later
    cH0 = CosConst.cH70

    z_model = np.arange(0.001, 3.0, 0.005)
    mu_model = get_mu(z_model, cosmo)

    ifn = interp1d(mu_model, z_model)

    return ifn(mu_list)


def no_big_bang(Om): # For Omega_m-Omega_Lambda
     return 1 - Om + (3*Om**(4./3.))/(2.*(1 + np.sqrt(1 - 2*Om) - Om)**(1./3.)) + (3*((1 + np.sqrt(1 - 2*Om) - Om)*Om**2)**(1./3.))/2.

assert np.isclose(no_big_bang(0.1), 27./20)
assert np.isclose(no_big_bang(0.5), 2.)

class CosConst():
    c = 299792458. # c / (100 km/s/Mpc) / (10 pc) = 299792458
    c100_Mpc = 2997.92458
    cH100 = 5.*np.log10(c)
    cH70 = cH100 - 5.*np.log10(0.7)
    Hinv100 = 9.77813 # Gyrs

    O_gammahh = 0.000024729 #2.469e-5 # Komatsu 2009
    O_radhh = (1. + 0.2271*3.04)*O_gammahh # Komatsu 2009
    O_bhh = 0.02258 # Komatsu 2009, note that marginalizing over z_star is similar to marginalizing over O_bhh, so sometimes okay to fix

