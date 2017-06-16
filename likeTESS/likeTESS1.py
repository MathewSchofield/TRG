"""
load data from getData/
change the bandpass, the data length, remove noise, add TESS noise to make
'limit' spectrum
"""

import warnings
warnings.simplefilter(action = "ignore")
import numpy as np
import pandas as pd
import os
import glob
import sys
import timeit

TRG = os.getcwd().split('likeTESS')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset




if __name__ == "__main__":
    start = timeit.default_timer()

    # the time-series file directories, epic numbers & parameters
    ts = glob.glob(TRG + 'GetData' + os.sep + '20Stars' + os.sep + '*.dat')
    epic = [x.split('kplr')[1].split('_llc')[0] for x in ts]
    params = pd.read_csv(TRG + 'GetData' + os.sep + '20Stars' + os.sep + '20stars.csv')

    for i, fdir in enumerate(ts):

        star = Dataset(epic[i], fdir)  # create the object
        info = params[params['EPIC']==int(epic[i])]  # info on the object

        # NOTE: units of time are seconds. CHANGE Kp TO Ic MAGNITUDES!
        noise = star.calc_noise(imag=info['kic_kepmag'].as_matrix(), exptime=30.*60., teff=info['Teff'].as_matrix())
        noise = noise*1e6  # total noise in units of ppm

        # length: days. Kepler FFI cadence = 30 mins (48 observations per day)
        star.read_timeseries(start=0, length=27*48, bandpass=0.85, noise=noise)
        star.plot_timeseries()



        #print star.time
        #print star.time_fix
        #print star.flux
        #print star.flux_fix

        # length: days
        #star.power_spectrum(start=0, length=27, noise=noise, madVar=True)
        #star.plot_power_spectrum()
        #print star.time

        sys.exit()






    stop = timeit.default_timer()
    print stop-start, 'secs'




#
