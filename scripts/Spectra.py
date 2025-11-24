from scipy import interpolate
import numpy as np
import os
from FileRead import clean_lines
from FileRead import read_param, readcol


def get_ang(ra1, dec1, ra2, dec2):
    r1 = np.array([np.cos(ra1*np.pi/180.)*np.cos(dec1*np.pi/180.), np.sin(ra1*np.pi/180.)*np.cos(dec1*np.pi/180.), np.sin(dec1*np.pi/180.)], dtype=np.float64)
    r2 = np.array([np.cos(ra2*np.pi/180.)*np.cos(dec2*np.pi/180.), np.sin(ra2*np.pi/180.)*np.cos(dec2*np.pi/180.), np.sin(dec2*np.pi/180.)], dtype=np.float64)
    
    ang = np.arccos(np.dot(r1, r2))*180./np.pi
    return ang


def RA_Dec_to_focal_plane(RAs, Decs):
    P = np.array([3.64960466e+01, 1.50115249e+02, 2.14859099e+02, 3.33877844e+02,
               -4.49495043e+00, 2.21087566e+00, 5.26792496e+01, -1.77298322e+01,
               3.84718562e-02])

    field = np.array(np.around((RAs - 20.)/100.), dtype=np.int32)
    
    dx = (RAs - P[field])*np.cos(Decs/57.2957795)
    dy = (Decs - P[field + 4])
    dr = np.sqrt(dx**2. + dy**2.)/P[-1]
    return dr



def check_derivs(thepath, flname = "result_deriv.dat"):
    try:
        f = open(thepath + "/" + flname, 'r')
    except:
        print("Missing file, can't check!")
        return None
    
    lines = f.read()
    f.close()
        
    if lines.count("Check") != 0:
        # If anything is in the file at all
        lines = lines.split('\n')
        
        for line in lines:
            parsed = line.split(None)
            if parsed.count("Check") and parsed.count("All|All|All"):
                print(parsed)
                d_dzps = [float(item) for item in parsed[4:8]]
            
                if np.abs(np.log(d_dzps[0])) < 0.2 and np.abs(np.log(d_dzps[1])) < 0.2 and np.abs(d_dzps[2]) < 0.2 and np.abs(d_dzps[3]) < 0.2:
                    return True
                else:
                    return False
    else:
        print("Empty file, can't check!", thepath)
        return False



def read_weightmat(weightfl, flux_scale, single_offset):
    print("Reading weightmat", weightfl)
    
    f = open(weightfl)
    lines = f.read().split('\n')
    f.close()

    lines = clean_lines(lines)
    lines = [item.split(None) for item in lines]
    if len(lines[0]) == 2:
        del lines[0]

    wmat = np.zeros([len(lines)]*2, dtype=np.float64)
    for i in range(len(wmat)):
        for j in range(i, len(wmat)):
            wmat[i,j] = float(lines[i][j])
            wmat[j,i] = wmat[i,j]


    cmat = np.linalg.inv(wmat)
    cmat *= np.outer(flux_scale, flux_scale)
    
    if not single_offset or len(cmat) == 1:
        wmat = np.linalg.inv(cmat)
        return wmat, 0


    
    offdiags = []
    for i in range(len(cmat)):
        for j in range(i+1, len(cmat)):
            offdiags.append(np.sqrt(np.abs(cmat[i,j])))# abs just in case
    return 1./diag(cmat - mean(offdiags)**2.), 1./mean(offdiags)**2.
    """
    for i in range(len(cmat)):
        for j in range(len(cmat)):
            print "%.1f" % sqrt(cmat[i,j]),
        print
    """




def get_lc2data(lc2fl, global_zp = 25., single_offset = False):
    lc2data = {}

    lc2data["instrument"] = read_param(lc2fl, "@INSTRUMENT")
    lc2data["band"] = read_param(lc2fl, "@BAND")
    lc2data["magsys"] = read_param(lc2fl, "@MAGSYS")

    xpos = read_param(lc2fl, "@X_FOCAL_PLANE")
    ypos = read_param(lc2fl, "@Y_FOCAL_PLANE")

    if xpos != None:
        lc2data["radius"] = np.sqrt(xpos**2. + ypos**2.)
    else:
        lc2data["radius"] = None
    
    [date, flux, flux_err, flux_zp] = readcol(lc2fl, 'ffff')

    if len(flux) == 0:
        [date, mag, mag_err] = readcol(lc2fl, 'fff')
        flux_zp = np.ones(len(date), dtype=np.float64)*global_zp
        flux = 10.**(0.4*(flux_zp - mag))
        flux_err = mag_err*flux*np.log(10.)/2.5

    flux_scale = 10.**(0.4*(global_zp - flux_zp))
    flux *= flux_scale
    flux_err *= flux_scale

    lc2data["flux"] = flux
    lc2data["date"] = date

    weight_fl = read_param(lc2fl, "@WEIGHTMAT")

    if weight_fl != None:
        # If single_offset == True, this will be 1D weights and an offset weight.
        # If single_offset == False, this will be a weight matrix and zero.

        if lc2fl.count("/"):
            weight_prefix = lc2fl[:lc2fl.rfind("/")] + "/"
        else:
            weight_prefix = ""
            
        weights, offset = read_weightmat(weight_prefix + weight_fl,
                                         flux_scale,
                                         single_offset)
        lc2data["weight"] = weights
        lc2data["offset"] = offset
    else:
        lc2data["weight"] = 1./flux_err**2.
        lc2data["offset"] = 0.

    return lc2data


def CCM(wave, R_V): # A(wave)/A_V
    x = 1./(wave/10000.) # um^-1
    y = (x - 1.82)



    # 0.3 to 1.1
    a = (0.3 <= x)*(x <= 1.1)*0.574*(x**1.61)
    b = (0.3 <= x)*(x <= 1.1)*(-0.527*x**1.61)


    a += (1.1 < x)*(x <= 3.3)*(1. + 0.17699*y - 0.50447*y**2. - 0.02427*y**3. + 0.72085*y**4. + 0.01979*y**5. - 0.77530*y**6. + 0.32999*y**7.)
    b += (1.1 < x)*(x <= 3.3)*(1.41338*y + 2.28305*y**2. + 1.07233*y**3. - 5.38434*y**4. - 0.62251*y**5. + 5.30260*y**6. - 2.09002*y**7.)


    
    Fa = (x >= 5.9)*(-0.04473*(x - 5.9)**2. - 0.009779*(x - 5.9)**3.)
    Fb = (x >= 5.9)*(0.2130*(x - 5.9)**2. + 0.1207*(x - 5.9)**3.)

    
    a += (3.3 < x)*(x <= 8.)*(1.752 - 0.316*x - 0.104/((x - 4.67)**2. + 0.341) + Fa)
    b += (3.3 < x)*(x <= 8.)*(-3.090 + 1.825*x + 1.206/((x - 4.62)**2. + 0.263) + Fb)


    a += (8. < x)*(x <= 10.)*(-1.073 - 0.628*(x - 8.) + 0.137*(x - 8.)**2. - 0.070*(x - 8.)**3.)
    b += (8. < x)*(x <= 10.)*(13.670 + 4.257*(x - 8.) - 0.420*(x - 8.)**2. + 0.374*(x - 8.)**3.)
    

    return a + b/R_V


def file_to_function(file_path, fill_value = "nan", verbose = True, normalize = "max"):

    f = open(file_path)
    lines = f.read()
    f.close()

    lines = lines.split('\n')
    lines = clean_lines(lines)

    spectrum = []

    for line in lines:
        parsed = line.split(None)
        try:
            spectrum.append([float(parsed[0]), float(parsed[1])])
        except:
            if verbose:
                print("Skipping Line ", line)
            else:
                pass

    spectrum = np.array(spectrum, dtype=np.float64)

    if normalize == "max":
        spectrum[:, 1] /= np.max(spectrum[:, 1])
    elif normalize != None:
        spectrum[:, 1] *= normalize

    interp_function = interpolate.interp1d(spectrum[:, 0], spectrum[:, 1], kind = 'linear', bounds_error = False, fill_value = fill_value)
    return interp_function


def get_mag_offset(file, instrument, band):
    f = open(file)
    lines = f.read().split('\n')
    f.close()
    
    lines = clean_lines(lines)
    lines = [item.split(None) for item in lines]
    
    for line in lines:
        if line[0] == instrument and line[1] == band:
            return float(line[2])


def radectoxyz(RAdeg, DECdeg):
    x = np.cos(DECdeg/(180./np.pi))*np.cos(RAdeg/(180./np.pi))
    y = np.cos(DECdeg/(180./np.pi))*np.sin(RAdeg/(180./np.pi))
    z = np.sin(DECdeg/(180./np.pi))

    return np.array([x, y, z], dtype=np.float64)

def radecztoxyzMpc(RAdeg, Decdeg, z):
    return 4282.7494*z*radectoxyz(RAdeg, Decdeg)


def get_dz(RAdeg, DECdeg):
    
    dzCMB = 371.e3/299792458. # NED
    #http://arxiv.org/pdf/astro-ph/9609034
    #CMBcoordsRA = 167.98750000 # J2000 Lineweaver
    #CMBcoordsDEC = -7.22000000
    CMBcoordsRA = 168.01190437 # NED
    CMBcoordsDEC = -6.98296811
          

    CMBxyz = radectoxyz(CMBcoordsRA, CMBcoordsDEC)
    inputxyz = radectoxyz(RAdeg, DECdeg)
    
    dz = dzCMB*np.dot(CMBxyz, inputxyz)
    dv = dzCMB*np.dot(CMBxyz, inputxyz)*299792.458

    print("Add this to z_helio to lowest order:")
    print(dz, dv)

    return dz




def get_zCMB(RAdeg, DECdeg, z_helio):
    dz = -get_dz(RAdeg, DECdeg)

    one_plus_z_pec = np.sqrt((1. + dz)/(1. - dz))
    one_plus_z_CMB = (1 + z_helio)/one_plus_z_pec
    return one_plus_z_CMB - 1.

def get_zhelio(RAdeg, DECdeg, z_CMB):
    dz = -get_dz(RAdeg, DECdeg)

    one_plus_z_pec = np.sqrt((1. + dz)/(1. - dz))
    one_plus_z_helio = (1 + z_CMB)*one_plus_z_pec
    return one_plus_z_helio - 1.



class Spectra:
    import os

    def __init__(self, band, instrument, obslambdas = None, pathmodel = os.environ["PATHMODEL"], radialpos = None, magsys = None):
        self.pathmodel = pathmodel
        self.band = band
        self.instrument = instrument
        self.obslambdas = obslambdas
        self.radialpos = radialpos

        self.magsys = magsys
        if magsys != None:
            if magsys[0] != "@":
                self.magsys = "@" + magsys
            
        self.get_band()

    def get_rad_filter(self, directory_path, filterwheel):
        f = open(directory_path + filterwheel)
        lines = clean_lines(   f.read().split('\n')   )
        f.close()

        filter_radpos_list = []
        for line in lines:
            parsed = line.split(None)
            if parsed[0] == self.band:
                radialpos = read_param(directory_path + parsed[-1], "@MEASUREMENT_RADIUS", ind = 1)
                filter_radpos_list.append([directory_path + parsed[-1], radialpos])
        filter_radpos_list.sort()

        filter_list = [item[0] for item in filter_radpos_list]
        radial_list = [item[1] for item in filter_radpos_list]

        if self.radialpos <= radial_list[0]:
            return file_to_function(filter_list[0], fill_value = 0.)
        if self.radialpos >= radial_list[-1]:
            return file_to_function(filter_list[-1], fill_value = 0.)

        radial_list = np.array(radial_list)

        inds = np.argsort(np.abs(radial_list - self.radialpos))
        
        f0 = file_to_function(filter_list[inds[0]], fill_value = 0.)
        f1 = file_to_function(filter_list[inds[1]], fill_value = 0.)

        x0 = radial_list[inds[0]]
        x1 = radial_list[inds[1]]

        return lambda x: (f0(x)*self.radialpos - f1(x)*self.radialpos + f1(x)*x0 - f0(x)*x1)/(x0 - x1)

        
    def get_band(self):

        # Get path to directory with filters:
        directory_path = self.pathmodel + "/" + read_param(self.pathmodel + "/fitmodel.card", "@" + self.instrument, 0) + "/"
        
        print("[instrument, directory_path, band] ", [self.instrument, directory_path, self.band])
        instrument_cards = directory_path + "instrument.cards"

        optics_trans_path = read_param(instrument_cards, "@OPTICS_TRANS")
        mirror_reflect_path = read_param(instrument_cards, "@MIRROR_REFLECTIVITY")
        atm_trans_path = read_param(instrument_cards, "@ATMOSPHERIC_TRANS")
        qe_path = read_param(instrument_cards, "@QE")

        print("self.radialpos", self.radialpos)
        if self.radialpos == None:
            filters_path = directory_path + read_param(instrument_cards, "@FILTERS")
            print("filters_path", filters_path)
            
            try:
                this_filter_path = directory_path + read_param(filters_path, self.band, ind = 2)
            except:
                this_filter_path = directory_path + read_param(filters_path, self.band, ind = 1)
            transmission_fn = file_to_function(this_filter_path, fill_value = 0.)

        else:
            print("Radial filter found!", self.radialpos)
            filterwheel = read_param(instrument_cards, "@RADIALLY_VARIABLE_FILTERS")
            transmission_fn = self.get_rad_filter(directory_path, filterwheel)



        
        if True:
            try:
                print("Trying @PSF_CORRECTION_FILTERS")
                psf_correction_filters_path = directory_path + read_param(instrument_cards, "@PSF_CORRECTION_FILTERS")
                this_psf_correction_filter_path = directory_path + read_param(psf_correction_filters_path, self.band, ind = 1)
                psf_correction = file_to_function(this_psf_correction_filter_path, fill_value = 0.)
                print("PSF Correction found for ", self.band)
            except:
                try:
                    print("Trying @CHROMATIC_CORRECTIONS")
                    psf_correction_filters_path = directory_path + read_param(instrument_cards, "@CHROMATIC_CORRECTIONS")
                    this_psf_correction_filter_path = directory_path + read_param(psf_correction_filters_path, self.band, ind = 1)
                    psf_correction = file_to_function(this_psf_correction_filter_path, fill_value = 0.)
                    print("PSF Correction found for ", self.band)
                except:
                    print("Setting PSF Correction to 1")
                    psf_correction = lambda x:1.
                
            if optics_trans_path != 1 and optics_trans_path != None:
                optics_fn = file_to_function(directory_path + optics_trans_path, fill_value = 0.)
            else:
                print("Setting optics to 1")
                optics_fn = lambda x:1.
                
            if mirror_reflect_path != 1 and mirror_reflect_path != None:
                mirror_fn = file_to_function(directory_path + mirror_reflect_path, fill_value = 0.)
            else:
                print("Setting mirror to 1")
                mirror_fn = lambda x:1.

            if atm_trans_path != 1 and atm_trans_path != None:
                atm_fn = file_to_function(directory_path + atm_trans_path, fill_value = 0.)
            else:
                print("Setting atmosphere to 1")
                atm_fn = lambda x:1.

            if qe_path != 1 and qe_path != None:
                qe_fn = file_to_function(directory_path + qe_path, fill_value = 0.)
            else:
                print("Setting QE to 1")
                qe_fn = lambda x:1.

        self.transmission_fn = lambda x: transmission_fn(x)*psf_correction(x)*optics_fn(x)*mirror_fn(x)*atm_fn(x)*qe_fn(x)

        if np.all(self.obslambdas != None):
            self.evaluated = self.transmission_fn(self.obslambdas)

        if self.magsys != None:
            print("Reading magsys...")
        
            if self.magsys == "@AB":
                self.ref_fn = file_to_function(self.pathmodel + "/MagSys/ab-spec.dat",
                                               fill_value = 0., normalize = 1.)
            else:
                magsysfile = read_param(self.pathmodel + "/fitmodel.card", self.magsys)
                mag_offset = get_mag_offset(self.pathmodel + "/" +  magsysfile, self.instrument, self.band)

                print("mag_offset ", self.instrument, self.band, mag_offset)
                self.ref_fn = file_to_function(self.pathmodel + "/" + read_param(self.pathmodel + "/" + magsysfile, "@SPECTRUM"),
                                               fill_value = 0., normalize = 10.**(0.4*mag_offset))


                # Let's note some signs. BD+17 on the Vega system is ~ 9, so the right sign is -2.5*log10(flux measured with the same zeropoint).
                # The flux of BD+17 here would be about 2.5e-4*Vega, so we would read in the BD+17 spectrum, then apply a normalization of 10.**(0.4*9).

                # What about AB offsets? Suppose SDSS_Mag - AB_Mag is found to be 0.03. Then the reference for SDSS_Mag is AB*1.03:
                # SDSS_Mag - AB_Mag = -2.5*log10(flux/(AB*1.03)) - -2.5*log10(flux/AB) = 0.03, so mag_offset should be +0.03.