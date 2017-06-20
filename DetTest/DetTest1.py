"""
- load data from GetData/
- make the data look like TESS from likeTESS/
- run a detection test on these power spectra
"""

import warnings
warnings.simplefilter(action = "ignore")
import numpy as np
import pandas as pd
import os
import glob
import sys
import timeit

TRG = os.getcwd().split('DetTest')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG + 'likeTESS' + os.sep)
from likeTESS1 import getInput


if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags = getInput(RepoLoc=TRG, dataset='20Stars')


    for i, fdir in enumerate(ts):

        star = Dataset(epic[i], fdir)  # create the object
        info = params[params['KIC']==int(epic[i])]  # info on the object
        mag = mags[mags['KIC']=='KIC ' + str(epic[i])]  # magnitudes from Simbad

        # units of exptime are seconds. noise in units of ppm
        star.calc_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
            e_lng=mag['e_lng'].as_matrix(), e_lat=mag['e_lat'].as_matrix(),\
            teff=info['Teff'].as_matrix())

        # make the data TESS-like in time domain before converting to frequency
        # Kepler FFI cadence = 30 mins (48 observations per day)
        #star.read_timeseries(start=0, length=27*48, bandpass=0.85)
        #star.plot_timeseries()
        #star.plot_power_spectrum()


        # convert from time to freq before making the data TESS-like. length: days
        star.power_spectrum(start=0, length=27, bandpass=0.85, madVar=True)
        star.plot_timeseries()
        star.plot_power_spectrum()

        sys.exit()


    stop = timeit.default_timer()
    print round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.'
