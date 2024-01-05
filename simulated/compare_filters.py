import sncosmo
import Spectra
import numpy as np


for item in [["sdssu", "SDSS", "SDSS_u"],
             ["sdssg", "SDSS", "SDSS_g"],
             ["sdssr", "SDSS", "SDSS_r"],
             ["sdssi", "SDSS", "SDSS_i"],
             ["sdssz", "SDSS", "SDSS_z"],
             ["f775w", "ACSWF", "F775W"],
             ["f850lp", "ACSWF", "F850LP"],
             ["f105w", "WFC3", "WFC3_f105w"],
             ["f125w", "WFC3", "WFC3_f125w"],
             ["f160w", "WFC3", "WFC3_f160w"]]:

    SNC = sncosmo.get_bandpass(item[0])
    U3 = Spectra.Spectra(instrument = item[1], band = item[2])
    
    SNC_waves = np.arange(SNC.minwave() - 100, SNC.maxwave() + 100, dtype=np.float64)
    SNC_vals = SNC(SNC_waves)
    SNC_weff = sum(SNC_vals*SNC_waves)/sum(SNC_vals)
    SNC_w2eff = sum(SNC_vals*(SNC_waves - SNC_weff)**2.)/sum(SNC_vals)

    U3_vals = U3.transmission_fn(SNC_waves)
    U3_weff = sum(U3_vals*SNC_waves)/sum(U3_vals)
    U3_w2eff = sum(U3_vals*(SNC_waves - U3_weff)**2.)/sum(U3_vals)

    print("%s, %s, %f, %f, %s, %.4f %.4f" % (item[0], "SNC", SNC_weff, np.sqrt(SNC_w2eff), "U3", np.log(U3_weff/SNC_weff), np.log(np.sqrt(U3_w2eff/SNC_w2eff))))
