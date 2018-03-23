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
import matplotlib.pyplot as plt
from scipy import stats

TRG = os.getcwd().split('likeTESS')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG)
from config import *  # the directories to find the data files


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

    # print mags
    # sys.exit()

    # if Imags are not known for the stars, calculate them and save to file
    if 'Imag' not in mags.columns:

        mags.rename(columns={'typed ident ':'KIC', 'Mag B ':'Bmag', \
            'Mag V ':'Vmag', '  coord3 (EclJ2000/2000)  ':'a'}, inplace=True)

        # ignore rows without Bmag (or vmag values)
        # mags = mags[(mags['Bmag'].str.strip()!='~')]
        # mags = mags[(mags['Vmag'].str.strip()!='~')]

        # do not delete rows without magnitudes, just set values to zero incase values become available
        # mags['Bmag'][(mags['Bmag'].str.strip()=='~')] = 0
        # mags['Vmag'][(mags['Vmag'].str.strip()=='~')] = 0
        # print mags
        # sys.exit()

        # separate ecliptic coordinates
        if 'a' in mags.columns:
            s = mags['a'].apply(lambda x: x.split(' +'))
            mags['e_lng'] = s.apply(lambda x: x[0]).astype(float)
            mags['e_lat'] = s.apply(lambda x: x[1]).astype(float)
            mags.drop(['a'], axis=1, inplace=True)

        # print mags
        # mags.to_csv(mag_file, index=False)
        # sys.exit()

        mags = BV2VI(mags)  # calculate Imags

        mags[['KIC', 'Bmag', 'Vmag', 'e_lng', 'e_lat', 'B-V', 'V-I', 'Imag']].\
            to_csv(mag_file, index=False)

    return ts, epic, params, mags, modes


def Plot1():
    """ Make a plot of numax(Teff) for the original 1000 Kepler Red Giants.
    Add a KDE to show density distribution. """

    plt.rc('font', size=14)
    fig, ax = plt.subplots()

    # metallicity: 0.0332/0.679=0.05, 0.009/0.732=0.01, 0.0023/0.746=0.003
    #tracks = glob.glob('/home/mxs191/Desktop/phd_y2/BenRendle_tracks/*.txt')
    # tracks = glob.glob('/home/mxs191/Desktop/phd_y2/BenRendle_tracks/*X0.746*.txt')
    # tracks = glob.glob('/home/mxs191/Desktop/phd_y2/BenRendle_tracks/*X0.679*.txt')
    tracks = glob.glob('/home/mxs191/Desktop/phd_y2/BenRendle_tracks/*X0.732*.txt')
    for idx, track_floc in enumerate(tracks):

        track = pd.read_csv(track_floc, sep='\s+', skiprows=2)

        track['teff'] = 10.**track['3:log(Te)'].as_matrix()  # Teff in Kelvin
        track['lum'] = 10.**track['4:log(L/Lo)'].as_matrix()  # Luminosity (solar units)
        track['rad'] = track['lum'].as_matrix()**0.5 * ((track['teff'].as_matrix()/5777.)**-2)  # radius (solar units)
        track['numax'] = 3090.*(track['rad'].as_matrix()**-1.85)*((track['teff'].as_matrix()/5777.)**0.92)  # mu Hz

        #print track[['3:log(Te)']]
        # print track[['teff', 'lum', 'rad', 'numax']]
        # sys.exit()
        plt.plot(track['teff'][43:], track['numax'][43:])

    # plt.xlim(7700,4300)
    # plt.ylim(0.3,50)
    # plt.yscale('log')
    # plt.show()
    # sys.exit()

    values = np.vstack([params['Teff'], params['numax']])
    kde_model = stats.gaussian_kde(values)  # the kernel
    params['kde'] = kde_model(values)  # the result of the kernel at these teffs and lums
    normfac = np.max(params['kde'])/0.99
    params['kde'] /= np.max(params['kde'])


    # trackloc = '/home/mxs191/Desktop/MSc/data_files/02.09 diegos solar-like tracks from sim 2/track text files/'
    # teff_08m, numax_08m = np.loadtxt(trackloc + 'm0.8.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_10m, numax_10m = np.loadtxt(trackloc + 'm1.0.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_12m, numax_12m = np.loadtxt(trackloc + 'm1.2.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_14m, numax_14m = np.loadtxt(trackloc + 'm1.4.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_16m, numax_16m = np.loadtxt(trackloc + 'm1.6.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_18m, numax_18m = np.loadtxt(trackloc + 'm1.8.txt', skiprows = 2, usecols = (2,7), unpack = True)
    # teff_20m, numax_20m = np.loadtxt(trackloc + 'm2.0.txt', skiprows = 2, usecols = (2,7), unpack = True)
    #
    # figt1 = plt.plot(teff_08m[940:5000], numax_08m[940:5000], color='k', linewidth=2.0)
    # figt2 = plt.plot(teff_10m[967:5000], numax_10m[967:5000], color='k', linewidth=2.0)
    # figt3 = plt.plot(teff_12m[985:5000], numax_12m[985:5000], color='k', linewidth=2.0)
    # figt3 = plt.plot(teff_14m[1040:5000], numax_14m[1040:5000], color='k', linewidth=2.0)
    # figt3 = plt.plot(teff_16m[1100:5000], numax_16m[1100:5000], color='k', linewidth=2.0)
    # figt3 = plt.plot(teff_18m[1090:5000], numax_18m[1090:5000], color='k', linewidth=2.0)
    # figt3 = plt.plot(teff_20m[1093:5000], numax_20m[1093:5000], color='k', linewidth=2.0)

    plt.scatter(params['Teff'], params['numax'], s=3, c=params['kde'])
    plt.colorbar(label='KDE')

    # bc = pd.read_csv('/home/mxs191/Desktop/MSc/data_files/Bright_catalogue_after_sav2csv2.py/all_bright_catalogue_data_test.csv')
    # print bc.shape
    # bc = bc[['teff', 'numax']].iloc[::10]
    # print bc.shape
    #plt.scatter(bc['teff'], bc['numax'], c='gray', s=1)

    plt.xlim(5500,4000)
    plt.ylim(400,2.7)
    plt.yscale('log')
    plt.xlabel(r'$T_{\textrm{eff}}$ / K')
    plt.ylabel(r'$\nu_{\rm max}$ / $\rm \mu Hz$')

    plt.show()

    sys.exit()

if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags, modes = getInput()
    Plot1()

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
