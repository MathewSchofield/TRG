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
import glob
import sys
import timeit

TRG = os.getcwd().split('likeTESS')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset


# all stars here are Red Giants so conditions for dwarfs have been removed
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
def getInput(RepoLoc, dataset):

    ts = glob.glob(RepoLoc + 'GetData' + os.sep + dataset + os.sep + '*.dat')
    epic = [x.split('kplr')[1].split('_llc')[0] for x in ts]
    params = pd.read_csv(RepoLoc + 'GetData' + os.sep + dataset + os.sep + dataset.lower() + '.csv')

    mags = pd.read_csv(RepoLoc + 'GetData' + os.sep + dataset + os.sep +\
        '20stars_simbad.csv', sep='\t', usecols=['typed ident ', 'Mag B ', 'Mag V ', '  coord3 (Ecl,J2000/2000)  '])
    mags.rename(columns={'typed ident ':'KIC', 'Mag B ':'Bmag', \
        'Mag V ':'Vmag', '  coord3 (Ecl,J2000/2000)  ':'a'}, inplace=True)
    mags = BV2VI(mags)  # calculate Imags

    # separate ecliptic coordinates
    s = mags['a'].apply(lambda x: x.split(' +'))
    mags['e_lng'] = s.apply(lambda x: x[0]).astype(float)
    mags['e_lat'] = s.apply(lambda x: x[1]).astype(float)
    mags.drop(['a'], axis=1, inplace=True)

    return ts, epic, params, mags


if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags = getInput(RepoLoc=TRG, dataset='20Stars')


    for i, fdir in enumerate(ts):

        star = Dataset(epic[i], fdir)  # create the object
        info = params[params['KIC']==int(epic[i])]  # info on the object
        mag = mags[mags['KIC']=='KIC ' + str(epic[i])]  # magnitudes from Simbad

        # units of exptime are seconds. noise in units of ppm
        noise = star.calc_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
            e_lng=mag['e_lng'].as_matrix(), e_lat=mag['e_lat'].as_matrix(),\
            teff=info['Teff'].as_matrix())

        # make the data TESS-like in time domain before converting to frequency
        # Kepler FFI cadence = 30 mins (48 observations per day)
        star.read_timeseries(start=0, length=27*48, bandpass=0.85, noise=noise)
        #star.plot_timeseries()
        star.plot_power_spectrum()


        # convert from time to freq before making the data TESS-like. length: days
        #star.power_spectrum(start=0, length=27, noise=noise, bandpass=0.85, madVar=True)
        #star.plot_timeseries()
        #star.plot_power_spectrum()

        sys.exit()


    stop = timeit.default_timer()
    print round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.'




#
