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
import matplotlib.pyplot as plt

TRG = os.getcwd().split('DetTest')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG + 'likeTESS' + os.sep)
from likeTESS1 import getInput


if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags = getInput(RepoLoc=TRG, dataset='20Stars')


    for i, fdir in enumerate(ts):

        star = Dataset(epic[i], fdir, bandpass=0.85, Tobs=27)  # Tobs in days
        info = params[params['KIC']==int(epic[i])]  # info on the object
        mag = mags[mags['KIC']=='KIC ' + str(epic[i])]  # magnitudes from Simbad

        # units of exptime are seconds. noise in units of ppm
        star.TESS_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
            teff=info['Teff'].as_matrix(), e_lat=mag['e_lat'].as_matrix(), sys_limit=0)
        star.kepler_noise(Kp=info['kic_kepmag'].as_matrix())


        # make the data TESS-like in time domain before converting to frequency
        star.timeseries(plot_ts=True, plot_ps=True)

        # convert from time to freq before making the data TESS-like
        star.power_spectrum(plot_ts=False, plot_ps=True)
        sys.exit()

        # make the original Kepler PS
        #star.ts()
        #star.Periodogram()
        #star.plot_power_spectrum()



    stop = timeit.default_timer()
    print round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.'
