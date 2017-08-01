"""
- load data from getData/
- change the bandpass, the data length, remove noise, add TESS noise to make
'limit' spectrum
"""

import warnings
warnings.simplefilter(action = "ignore")
import numpy as np
import pandas as pd
import os
import pwd
import glob
import sys
import timeit
from config import *

TRG = os.getcwd().split('likeTESS')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
user = pwd.getpwuid(os.getuid())[0]
if user == 'davies':
    TRG = '/home/davies/Projects/Mat/TRG/'
    data_dir = '/home/davies/Dropbox/K2_seismo_pipes/20Stars/Data/'
elif user == 'Mat':
    pass
    

# calculate Imags. all stars here are Red Giants so conditions for dwarfs have been removed
def BV2VI(whole):

    whole['B-V'] = whole['Bmag'] - whole['Vmag']
    whole = whole[(whole['B-V'] > -0.4) & (whole['B-V'] < 1.7)]
    #print whole.shape, 'after B-V cuts'

    whole['V-I'] = 100. # write over these values
    cg = [-0.8879586e-2, 0.7390707, 0.3271480, 0.1140169e1, -0.1908637, -0.7898824,
    	0.5190744, 0.5358868]

    # calculate (V-I) for giants
    x = whole['B-V'] - 1
    y = (cg[0] + cg[1]*x + cg[2]*(x**2) + cg[3]*(x**3) + cg[4]*(x**4) +\
    	cg[5]*(x**5) + cg[6]*(x**6) + cg[7]*(x**7))
    whole['V-I'] = y + 1
    x, y = [[] for i in range(2)]

    whole['Imag'] = whole['Vmag']-whole['V-I']
    return whole


# load the time-series file directories, epic numbers & parameters.
def getInput():

    # ts: time series file locs. epic: EPIC numbers. params: numax, dnu, teff, Kp
    # modes: list of mode frequencies, linewidths and uncertainties for each star
    ts = glob.glob(data_dir + '*.dat')
    epic = [x.split('kplr')[1].split('_llc')[0] for x in ts]
    params = pd.read_csv(param_file)
    modes = glob.glob(mode_dir + '*.csv')
    mags = pd.read_csv(mag_file)

    # if Imags are not known for the stars, calculate them and save to file
    if 'Imag' not in mags.columns:

        mags.rename(columns={'typed ident ':'KIC', 'Mag B ':'Bmag', \
            'Mag V ':'Vmag', '  coord3 (Ecl,J2000/2000)  ':'a'}, inplace=True)

        # separate ecliptic coordinates
        s = mags['a'].apply(lambda x: x.split(' +'))
        mags['e_lng'] = s.apply(lambda x: x[0]).astype(float)
        mags['e_lat'] = s.apply(lambda x: x[1]).astype(float)
        mags.drop(['a'], axis=1, inplace=True)

        mags = BV2VI(mags)  # calculate Imags

        mags[['KIC', 'Bmag', 'Vmag', 'e_lng', 'e_lat', 'B-V', 'V-I', 'Imag']].\
            to_csv(mag_file, index=False)

    return ts, epic, params, mags, modes


if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags, modes = getInput()

    for i, fdir in enumerate(ts):

        star = Dataset(epic[i], fdir, sat='Kepler', bandpass=0.85, Tobs=27)  # Tobs in days
        info = params[params['KIC']==int(epic[i])]  # info on the object
        mag = mags[mags['KIC']=='KIC ' + str(epic[i])]  # magnitudes from Simbad


        star.Diagnostic(Kp=info['kic_kepmag'].as_matrix(), \
            imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
            teff=info['Teff'].as_matrix(), e_lat=mag['e_lat'].as_matrix())
        sys.exit()

        # units of exptime are seconds. noise in units of ppm
        star.TESS_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
            teff=info['Teff'].as_matrix(), e_lat=mag['e_lat'].as_matrix(), sys_limit=0)
        star.kepler_noise(Kp=info['kic_kepmag'].as_matrix())


        # make the data TESS-like in time domain before converting to frequency
        star.timeseries(plot_ts=False, plot_ps=False)

        # convert from time to freq before making the data TESS-like
        star.power_spectrum(plot_ts=False, plot_ps=False)
        sys.exit()

        # make the original Kepler PS
        #star.ts()
        #star.Periodogram()
        #star.plot_power_spectrum()


    stop = timeit.default_timer()
    print(round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.')




#
