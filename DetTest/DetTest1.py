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
import scipy.ndimage as ndim
from scipy import stats
import timeit
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve

TRG = os.getcwd().split('DetTest')[0]
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG + 'likeTESS' + os.sep)
from likeTESS1 import getInput


class DetTest(object):

    def __init__(self, ds):
        """ Inherit from the Dataset class
        ds: an instance of the Dataset class """
        self.ds = ds
        self.bkg = []

    def test(self):
        print 'hg'
        print self.ds.epic

    def estimate_background(self, log_width=0.1):
        """ Estimate background of star using log-space filter
        Credit: DFM """

        count = np.zeros(len(self.ds.freq))
        bkg = np.zeros_like(self.ds.freq)  # array for the background
        x0 = np.log10(self.ds.freq[1])  # exponent to describe 1st bin

        while x0 < np.log10(self.ds.freq[-1]):  # until the end of the spectrum

            # bins to calculate bkg for (size increases with larger frequencies)
            m = np.abs(np.log10(self.ds.freq) - x0) < log_width

            # bkg is the mean power in the 'm' bins
            bkg[m] += np.median(self.ds.power[m]) * 1.4  # 1.4: convert from median to mean
            count[m] += 1  # normalise over the number of times this bin is used to get bkg
            x0 += 0.5*log_width  # iterate through the bins

        bkg[0] = self.ds.power[0]
        count[0] = 1 # the normalised background in the 1st bin is just the power
        return bkg/count

    def Power2SNR(self, plt_PS=False, plt_SNR=False):
        """ Convert from Power (ppm^2 muHz^-1) to SNR
        by dividing the signal by the fitted background """

        self.bkg = self.estimate_background(log_width=0.1)
        self.snr = self.ds.power/self.bkg

        if plt_PS:
            self.plot_ps()

        if plt_SNR:
            self.plot_snr()

    def plot_ps(self, smoo=0, plog=True):
        ''' Plots the power spectrum and the fitted background '''

        if len(self.ds.freq) < 0:
            self.ds.power_spectrum()

        fig, ax = plt.subplots()
        ax.plot(self.ds.freq, self.ds.power, 'k-', alpha=0.5)
        ax.plot(self.ds.freq, self.bkg, 'k-')

        if plog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        ax.set_ylabel(r'Power ($\rm ppm^{2} \, \mu Hz^{-1}$)')
        ax.set_xlim([self.ds.freq.min(),self.ds.freq.max()])
        ax.set_title('KIC ' + str(self.ds.epic))

        if smoo > 0:
            self.ds.rebin_quick(smoo)
            ax.plot(self.ds.smoo_freq, self.ds.smoo_power, 'k-', linewidth=4)

        plt.show()
        fig.savefig('ps_' + str(self.ds.epic) + '.png')

    def plot_snr(self, plog=True):
        ''' Plots the SNR '''
        if len(self.ds.freq) < 0:
            self.ds.power_spectrum()

        fig, ax = plt.subplots()
        ax.plot(self.ds.freq, self.snr, 'k-', alpha=0.5)
        if plog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        ax.set_ylabel(r'Power ($\rm ppm^{2} \, \mu Hz^{-1}$)')
        ax.set_xlim([self.ds.freq.min(),self.ds.freq.max()])
        ax.set_title('KIC ' + str(self.ds.epic))

        plt.show()
        fig.savefig('snr_' + str(self.ds.epic) + '.png')



if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags, modes = getInput(RepoLoc=TRG, dataset='20Stars')

    for i, fdir in enumerate(ts):

        ds = Dataset(epic[i], fdir, bandpass=0.85, Tobs=27)  # Tobs in days
        info = params[params['KIC']==int(epic[i])]  # info on the object
        mag = mags[mags['KIC']=='KIC ' + str(epic[i])]  # magnitudes from Simbad
        IDfile = [ID for ID in modes if ds.epic in ID][0]  # mode ID file loc
        ds.get_modes(IDfile)

        #print ds.mode_id
        #print len(ds.mode_id)
        #sys.exit()


        # make the original Kepler PS
        ds.ts()
        ds.Periodogram()
        #ds.plot_power_spectrum()

        star = DetTest(ds)
        star.Power2SNR(plt_PS=False, plt_SNR=False)
        #print star.snr


        snrs = np.full(len(ds.mode_id), -99)  # SNR values at radial mode freqs
        for idx, f in ds.mode_id.iterrows():

            #print abs(f['w0']), int(np.around(abs(f['w0'])))
            #print f['f0']

            snrs[idx] = star.snr[f['f0']]

            """
            # smooth with uniform filter
            smoo = ndim.filters.uniform_filter1d(star.snr, size=int(np.around(abs(f['w0']))))

            # smooth by convolving with Guassian
            g = Gaussian1DKernel(stddev=abs(f['w0']))
            smoo2 = convolve(star.snr, g, boundary='extend')

            # smooth by interpolating
            bins = np.arange(0., star.ds.freq[-1], abs(f['w0']))  # rebin data to get highest SNR
            smoo3 = np.interp(bins, star.ds.freq, star.snr)  # power values at these freqs
            #sys.exit()


            #print star.snr
            #star.snr_fix = smoo
            #print star.snr_fix

            #print star.snr
            print 'before smoo', star.snr[f['f0']]
            print 'smoo1', smoo[f['f0']]
            print 'smoo2', smoo2[f['f0']]
            print 'smoo3', smoo3[f['f0']], '\n'
            """

            #print smoo
            #star.plot_snr()
            #star.snr = smoo
            #star.plot_snr()



            #sys.exit()

        print snrs

        fap = 0.01  # false alarm probability
        pdet = 1.0 - fap
        nbins=1

        #snrthresh = stats.chi2.ppf(pdet, 2.0*nbins) / (2.0*nbins) - 1.0
        snrthresh = 1  # SNR value if no mode present
        prob = stats.chi2.sf((snrthresh+1.0) / (snrs+1.0)*2.0*nbins, 2*nbins) # detection probability for each mode
        print prob
        print snrthresh
        sys.exit()




        # units of exptime are seconds. noise in units of ppm
        #star.TESS_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
        #    teff=info['Teff'].as_matrix(), e_lat=mag['e_lat'].as_matrix(), sys_limit=0)
        #star.kepler_noise(Kp=info['kic_kepmag'].as_matrix())
        #star.timeseries(plot_ts=False, plot_ps=False)


    stop = timeit.default_timer()
    print round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.'



#
