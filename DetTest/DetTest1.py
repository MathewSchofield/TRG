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
from scipy.optimize import curve_fit
from scipy import stats
import timeit
import matplotlib.pyplot as plt
from astropy.convolution import Gaussian1DKernel, convolve
from scipy import interpolate

TRG = os.getcwd().split('DetTest')[0]
sys.path.insert(0, TRG)
from plotTemplates import generalPlot
from config import *  # the directories to find the data files
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
        self.snr = []
        self.snr_modes = np.array([])  # the self.snr values at mode frequencies
        self.prob = []

        # the location of the probability file for the star (to save) in Info2Save()
        self.probfile = os.getcwd() + os.sep + 'DetTest1_results' +\
            os.sep + self.ds.epic + '.csv'

    def estimate_background(self, log_width):
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
        """ Convert from Power (ppm^2 muHz^-1) to SNR by dividing the signal
        by the fitted background. If the power in the first bin is zero, change
        the SNR value in the first bin. """

        self.bkg = self.estimate_background(log_width=0.1)
        self.snr = self.ds.power/self.bkg
        if np.isnan(self.snr[0]) == True:  self.snr[0] = 1e-5

        if plt_PS:   self.plot_ps()
        if plt_SNR:  self.plot_snr()

    def set_width(self, numax, a=0.66, b=0.88, factor=1.5):
        """ Created by GD in K2pipes/K2wavelets.py """
        return a * numax**b * factor

    def get_snr(self, smoo=0, skips=50, plt_PS=False, plt_SNR=False):
        """ Created by GD in K2pipes/K2wavelets.py
        Takes a moving median around the predicted envelope width at every
        frequency. Then interpolates between median values."""

        # print np.abs(self.ds.freq - 100.)  # the difference between all the freqs and d
        # print self.set_width(100., factor=1)  # the env width at d
        # print self.ds.power[np.abs(self.ds.freq - 100.) < self.set_width(100., factor=1)]  # all the power values inside of the envelope around d
        # print np.median(self.ds.power[np.abs(self.ds.freq - 100.) < self.set_width(100., factor=1)])  # the median power in the envelope around d
        med = [np.median(self.ds.power[np.abs(self.ds.freq - d) < self.set_width(d, factor=1)]) for d in self.ds.freq[::skips]]

        # interpolate between skipped freqs in self.ds.freqs using the moving median
        f = interpolate.interp1d(self.ds.freq[::skips], med, bounds_error=False)
        self.bkg = f(self.ds.freq)
        self.snr = self.ds.power / self.bkg
        self.snr[:skips] = 1.0
        self.snr[-skips:] = 1.0
        if smoo > 1:
            self.snr = nd.filters.uniform_filter1d(self.snr, int(smoo[0]))

        if plt_PS:   self.plot_ps()
        if plt_SNR:  self.plot_snr()

    def mode_snrs(self, v=False):
        """ Get the SNR values for the modes by taking the highest value
        around the range defined by the mode linewidth.

        Inputs
        ds.mode_id:     frequency and linewidth values of the modes
        self.ds.freq:   frequency array of the spectrum
        self.snr_modes: empty array to put mode SNR values in (defined in __init__)

        Outputs
        self.snr_modes: full array of mode SNR values
        """

        # iterate over the fitted modes
        for idx, f in ds.mode_id.iterrows():

            #smoo = nd.filters.uniform_filter1d(self.snr, int(np.exp(f['w0'])/ds.bin_width))
            #smoo = self.Conv(self.snr, np.exp(f['w0'])/ds.bin_width)  # smooth the SNR by convolving with Guassian
            #smoo = self.Conv(self.snr, abs(f['w0']))  # smooth the SNR by convolving with Guassian
            #index = np.abs(self.ds.freq-f['f0']).argmin()  # frequency closest to mode
            #self.snr_modes = np.append(self.snr_modes, smoo[index])  # add the SNR value at the mode to the array

            # the range to find highest snr over
            wid = np.exp(f['w0'])
            rng = (self.ds.freq>(f['f0']-wid)) & (self.ds.freq<(f['f0']+wid))

            if v:  print f['f0'], wid
            if v:  print self.ds.freq[rng]
            if v:  print self.snr[rng]
            if v:  print max(self.snr[rng])  # the maximum SNR around the mode

            # if there are no freq bins in the range defined by mode width,
            # take closest snr value
            if len(self.ds.freq[rng]) == 0:
                index = np.abs(self.ds.freq-f['f0']).argmin()
                self.snr_modes = np.append(self.snr_modes, self.snr[index])

            else:
                self.snr_modes = np.append(self.snr_modes, max(self.snr[rng]))

            if v:  print 'final val', self.snr_modes, '\n'
            if v:  sys.exit()

    def Det_Prob(self, nbins=[], fap=[], snrthresh=[]):
        """
        Calculate the detection probability, given a SNR ratio and threshold.

        Inputs
        snrs:      Array of SNR values to calculate detection probabilities for (Float)
        nbins:     Number of bins the SNR is taken across (Int)
        fap:       False Alarm Probability (Float)
        snrthresh: Threshold SNR; SNR if no mode is present (Float)

        Outputs
        prob: (int:0 or 1) the detection probability for each SNR value (i.e mode)
        """

        if nbins == []:      nbins = 1
        if fap == []:        fap = 0.01
        if snrthresh == []:
            pdet = 1.0 - fap
            snrthresh = stats.chi2.ppf(pdet, 2.0*nbins) / (2.0*nbins) - 1.0

        #print self.snr_modes, snrs
        #sys.exit()
        self.prob = stats.chi2.sf((snrthresh+1.0) / (self.snr_modes+1.0)*2.0*nbins, 2*nbins)

        self.prob[self.prob>=0.9] = 1  # if Pdet>0.9, mode is detected
        self.prob[self.prob<0.9]  = 0  # if Pdet<0.9, mode is not detected

    def Conv(self, data, stddev):
        """
        Perform a convolution using a 1D Gaussian Kernel

        Inputs
        data:   data to perform convolution upon (array)
        stddev: standard deviation of the Gaussian to convolve with (Float)

        Outputs
        data:   data which has been convolved (array)
        """

        g = Gaussian1DKernel(stddev=stddev)
        data = convolve(data, g, boundary='extend')
        return data

    def Info2Save(self):
        """
        THIS FUNCTION IS DEPRECIATED. USE THE data_for_ML().

        Save information on mode frequencies, SNR values and detection
        probabilities for different satellites and observing times.

        Output
        Files saved in DetTest/DetTest1_results/ (.csv)
        """

        if self.ds.sat == 'Kepler':
            snr_header = 'SNR_' + self.ds.sat
            prob_header = 'Pdet_' + self.ds.sat

        elif self.ds.sat == 'TESS':
            snr_header = 'SNR_' + self.ds.sat + str(ds.Tobs)
            prob_header = 'Pdet_' + self.ds.sat + str(ds.Tobs)


        # check if file exists. If it does, add columns/write over columns
        # if the files doesn't exist, make it
        if os.path.exists(self.probfile):
            save = pd.read_csv(self.probfile)
            save[snr_header], save[prob_header] = [self.snr_modes, self.prob]

        else:
            save = pd.DataFrame({'f0'      :self.ds.mode_id['f0'],
                                snr_header :self.snr_modes,
                                prob_header:self.prob})
            save.sort_values(['f0'], axis=0, ascending=True, inplace=True)
            save = save.ix[:, ['f0', snr_header, prob_header]]

        save.to_csv(self.probfile, index=False)

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
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +'ps_' + str(self.ds.epic) + '.png')

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
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +'snr_' + str(self.ds.epic) + '.png')

    def Diagnostic_plot1(self, v=False):
        """
        Assess which method of convolution/interpolation gives the highest mode
        SNR values for 1 star
        """

        # sort modes by frequency (radial order)
        ds.mode_id.sort_values(['f0'], axis=0, ascending=True, inplace=True)

        # SNR values after smoothing/interpolating at radial mode freqs
        u  = np.full(len(ds.mode_id), -99)  # unsmoothed
        s1 = np.full(len(ds.mode_id), -99)  # after Gaussian smoothing
        s2 = np.full(len(ds.mode_id), -99)  # after uniform smoothing
        s3 = np.full(len(ds.mode_id), -99)  # after linear interpolation

        for idx, f in ds.mode_id.iterrows():
            width = abs(f['w0'])  # width to convolve/interpolate over

            # smooth by convolving with Guassian
            smoo = star.Conv(self.snr, width)

            # smooth with uniform filter
            smoo2 = ndim.filters.uniform_filter1d(self.snr, size=int(np.around(width)))

            # smooth by interpolating
            bins = np.arange(0., self.ds.freq[-1], width)  # rebin data to get highest SNR
            smoo3 = np.interp(bins, self.ds.freq, self.snr)  # SNR values at these freqs

            index = np.abs(self.ds.freq-f['f0']).argmin()  # use the frequency closest to mode
            if v:
                print(self.ds.freq[index], self.snr[index])
                print('before smoo', self.snr[index])
                print('smoo1', smoo[index])
                print('smoo2', smoo2[index])
                print('smoo3', smoo3[np.abs(bins-f['f0']).argmin()], '\n')

            u[idx]  = self.snr[index]
            s1[idx] = smoo[index]
            s2[idx] = smoo2[index]
            s3[idx] = smoo3[np.abs(bins-f['f0']).argmin()]

        fig = plt.figure(figsize=(12, 18))
        plt.rc('font', size=26)
        plt.plot(self.ds.mode_id['f0'], u,    label=r'unsmoothed')
        plt.plot(self.ds.mode_id['f0'], s1,   label=r'Smoothed with 1D Gaussian')
        plt.plot(self.ds.mode_id['f0'], s2,   label=r'Smoothed with uniform filter')
        plt.plot(self.ds.mode_id['f0'], s3,   label=r'Smoothed by interpolating')
        plt.xlabel(r'$\nu / \mu$Hz')
        plt.ylabel(r'SNR')
        plt.legend(loc='upper right')
        plt.show()
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +'DetTest_Diagnostic_plot1_' + self.ds.epic + '.pdf')
        #sys.exit()

    def Diagnostic_plot2(self):
        """ Compare detection probabilities and SNRs between different
        timeseries' and satellites for 1 star. """

        probs = pd.read_csv(self.probfile)

        fig, ax = generalPlot(xaxis=r'$\nu / \mu$Hz', yaxis=r'$P_{\rm det}$')
        plt.scatter(probs['f0'], probs['Pdet_Kepler'], label='Kepler - 4yrs')
        plt.scatter(probs['f0'], probs['Pdet_TESS365'], label='TESS - 1 yr')
        plt.scatter(probs['f0'], probs['Pdet_TESS27'], label='TESS - 27 days')
        plt.legend(loc='lower right')
        plt.ylim([0,1])
        plt.show()
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +\
            'DetTest_Diagnostic_plot2_Pdet' + self.ds.epic + '.pdf')

        fig, ax = generalPlot(xaxis=r'$\nu / \mu$Hz', yaxis=r'SNR')
        plt.scatter(probs['f0'], probs['SNR_Kepler'], label='Kepler - 4yrs')
        plt.scatter(probs['f0'], probs['SNR_TESS365'], label='TESS - 1 yr')
        plt.scatter(probs['f0'], probs['SNR_TESS27'], label='TESS - 27 days')
        plt.legend(loc='lower right')
        #plt.ylim([0,1])
        plt.show()
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +\
            'DetTest_Diagnostic_plot2_SNR' + self.ds.epic + '.pdf')

    def Diagnostic_plot3(self):
        """ Compare detection probabilities between different timeseries' and
        satellites for the entire dataset of stars.
        Plot results for 1 star at a time (the Pdet values for each mode).
        Also plot the median Pdet value from all modes and stars for each
        satellite and timeseries dataset. """

        floc = glob.glob(os.getcwd() + os.sep + 'DetTest1_results' + os.sep + '*.csv')
        fig, ax = generalPlot(xaxis=r'$\nu / \mu$Hz', yaxis=r'$P_{\rm det}$')

        for idx, i in enumerate(floc):

            d = pd.read_csv(i)
            if idx == 0:
                fullpdet = d[['f0', 'Pdet_Kepler', 'Pdet_TESS365', 'Pdet_TESS27']]
            else:
                fullpdet = pd.concat([ fullpdet,\
                    d[['f0', 'Pdet_Kepler', 'Pdet_TESS365', 'Pdet_TESS27']] ])

            plt.scatter(d['f0'], d['Pdet_Kepler'], color='b',\
                label="Kepler - 4yrs" if idx == 0 else '')
            plt.scatter(d['f0'], d['Pdet_TESS365'], color='orange',\
                label='TESS - 1 yr' if idx == 0 else '')
            plt.scatter(d['f0'], d['Pdet_TESS27'], color='g',\
                label='TESS - 27 days' if idx == 0 else '')

        plt.axhline(fullpdet['Pdet_Kepler'].median(), color='b')
        plt.axhline(fullpdet['Pdet_TESS365'].median(), color='orange')
        plt.axhline(fullpdet['Pdet_TESS27'].median(), color='g')

        plt.legend(loc='lower right')
        plt.ylim([0,1])
        plt.show()
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +\
            'DetTest_Diagnostic_plot3.pdf')
        sys.exit()

    def plot4(self, plog=False):
        """ Plot the SNR spectrum of the modes to check that the correct value
        of the SNR is being taken for each mode. """

        probs = pd.read_csv(self.probfile)

        fig, ax = plt.subplots()
        plt.plot(self.ds.freq, self.snr, 'k-', alpha=0.5)

        # plot the SNR range to search across when finding snr_modes
        for idx, line in enumerate(self.ds.mode_id['f0']):
            w = np.exp(self.ds.mode_id['w0'][idx])
            plt.axvline(x=line-w, color='b', linestyle='-', alpha=0.4)
            plt.axvline(x=line+w, color='b', linestyle='-', alpha=0.4)

        # overplot the predicted SNR values at the modes
        plt.scatter(probs['f0'], probs['SNR_Kepler'], label='Kepler - 4yrs')
        plt.scatter(probs['f0'], probs['SNR_TESS365'], label='TESS - 1 yr')
        plt.scatter(probs['f0'], probs['SNR_TESS27'], label='TESS - 27 days')

        if plog:
            plt.xscale('log')
            plt.yscale('log')
        plt.xlabel(r'Frequency ($\rm \mu Hz$)')
        plt.ylabel(r'SNR')

        mn = min(star.ds.mode_id['f0']) -\
            (max(star.ds.mode_id['f0'])-min(star.ds.mode_id['f0']))/7.
        mx = max(star.ds.mode_id['f0']) +\
            (max(star.ds.mode_id['f0'])-min(star.ds.mode_id['f0']))/7.
        plt.xlim([mn,mx])

        plt.legend()
        plt.title('KIC ' + str(self.ds.epic))
        plt.show()
        fig.savefig(os.getcwd() + os.sep + 'DetTest1_plots' + os.sep +\
            'plot4_SNR' + self.ds.epic + '.pdf')


class data_for_ML(object):
    """ Prepare and save the X, Y data to be used in ML.py """

    def __init__(self, star):
        self.star = star

    def Gauss(self, x, height, centre, width, b=0):
        """ A Gaussian fit.
            Inputs
            x:      data to fit Gaussian to
            height: the height of the curve's peak
            centre: the position of the centre of the peak
            width:  the RMS width of the curve
            b:      Optionally add background, b. """

        return height * np.exp(-(x - centre)**2 / (2 * width**2)) - b

    def fit_amplitudes(self, v=False):
        """ Fits the mode frequencies (x) and amplitudes (height) for each
        Kepler target to get a numax value. """

        x         = self.star.ds.mode_id['f0'].as_matrix()
        height    = self.star.ds.mode_id['A0'].as_matrix()
        sd_height = self.star.ds.mode_id['A0_err'].as_matrix()
        if v:  print 'x:', x
        if v:  print 'height:', height
        if v:  print 'sd_height:', sd_height

        mean = sum(height*x) / sum(height)  # weighted mean x value
        sigma = np.sqrt(sum(height * (x - mean)**2) / sum(height)) # weighted Gaussian sd

        popt, pcov = curve_fit(self.Gauss, x, height,\
            p0=[max(height), mean, sigma], sigma=sd_height, absolute_sigma=True)
        self.numax = popt[1]
        # sd_numax = np.sqrt(np.diag(pcov))[1]  # the uncertaintiy on numax
        if v:  print 'numax:', self.numax

    def average_dnu(self):
        """ get an average dnu value for each star. """

        freqs = self.star.ds.mode_id['f0'].as_matrix()
        f1 = freqs[1:]
        f2 = freqs[:-1]
        diff = f1-f2
        self.dnu = np.mean(diff)

    def save_xy(self, v=False, n=3):
        """ Save the X and Y data for Machine Learning into 1 file.
        x:            The number of iterations per star (Kp is varied each iteration).
                      Note: x*i+j is the row number of xy for every star.
        n:            The number of overtones (SNR/Pdet values) to save in the Y data.
        headers:      The headers to use in the final XY dataframe
        ML_data_dir:  (Defined in config.py) where to save the data
        xy:           The dataframe to save out for all stars. 1 row per star,
                      1 column per parameter
        """


        headers = ['KIC', 'numax', 'Dnu', 'Teff', '[M/H]2', 'kic_kepmag', 'Bmag',
                   'Vmag', 'B-V', 'V-I', 'Imag', 'Pdet1', 'Pdet2', 'Pdet3']

        #print info, mag

        if x*i+j == 0:
            """ On the first iteration, make the X and Y data arrays. """
            global x_data, y_data
            x_data = np.zeros((len(params)*x, 11))
            y_data = np.zeros((len(params)*x, n))


        # NOTE: Get X data.
        info['Dnu'] = self.dnu  # replace dnu values with the value from average_dnu()
        x_data[x*i+j, 0:6]  = info[headers[0:6]].as_matrix()
        x_data[x*i+j, 6:11] = mag[headers[6:11]].as_matrix()


        # NOTE: Get Y data.
        # Step 1: get the frequencies and their corresponding Pdet values
        ds.mode_id.reset_index(drop=True, inplace=True)
        if v:  print i
        if v:  print ds.mode_id['f0'].as_matrix()
        if v:  print star.snr_modes
        if v:  print star.prob, '\n'

        # step 2: calculate the frequency difference between the modes and numax
        diff = abs(float(info['numax']) - ds.mode_id['f0'].as_matrix())
        if v:  print float(info['numax']), diff, '\n'

        # step 3: take the 'n' modes closest to numax, sorted from highest to lowest SNR value
        # make sure that the mode closest to numax is at the centre of
        idx = np.argpartition(diff, n-1)[:n]
        if v:  print idx
        if v:  print ds.mode_id['f0'].as_matrix()[idx]
        if v:  print star.prob[idx], '\n'

        # step 4: sort the frequencies in ascending order. Sort SNR values to match the corresponding frequencies
        f = np.sort(ds.mode_id['f0'].as_matrix()[idx])
        p = ds.mode_id['f0'].as_matrix()[idx].argsort() # permutation that sorts the frequencies f
        s = star.prob[idx][p]
        y_data[x*i+j, :] = s  # put the sorted Pdet values into the Y data array
        if v:  print p
        if v:  print f[p], '\n'  # uncsorted frequencies
        if v:  print 'sorted freq:', f
        if v:  print 'sorted Pdet:', s, '\n'#, '\n', '\n'

        if (x*i+j)%100 == 0:  print i  # counter

        if i*(j+1) == (len(params)-1)*x:
            """ After the last star has been processed, save data for all stars. """

            d = np.concatenate((x_data, y_data), axis=1)  # put the x and y data into 1 array
            xy = pd.DataFrame(d, columns=headers)  # put x and y data into dataframe
            xy.to_csv(ML_data_dir + '_' + sat + '_XY.csv', index=False)  # ML_data_dir is defined in config.py)


if __name__ == "__main__":
    start = timeit.default_timer()

    ts, epic, params, mags, modes = getInput()

    for i, fdir in enumerate(ts):
        """ Loop through the timeseries files. 1 file (1 star) per iteration. """

        #sat = 'Kepler'
        sat = 'TESS'

        ds = Dataset(epic[i], fdir, sat=sat, bandpass=0.85, Tobs=27)  # Tobs in days
        info = params[params['KIC']==int(epic[i])]  # info on the object, for TESS_noise
        mag = mags[mags['KIC'].str.rstrip()=='KIC ' + str(epic[i])]  # magnitudes from Simbad


        """ Conditions to skip this star """
        if len(info) == 0:
            """ No APOKASC information given in 'params' file """
            #print 'No APOKASC info for KIC', ds.epic
            continue

        if [ID for ID in modes if ds.epic in ID] == []:
            """ No fitted mode file given for this star (in 'modes'), so it cannot be analysed. """
            #print 'No fitted mode file for KIC', ds.epic
            continue

        if len(mag) == 0:
            """ no magnitude values available for the star """
            continue

        IDfile = [ID for ID in modes if ds.epic in ID][0]  # mode ID file loc
        ds.get_modes(IDfile)
        #print epic[i]

        if len(ds.mode_id) == 0:
            """ length of mode id file is 0 for KIC """
            continue
        """ Conditions to skip this star """


        x = 100
        for j in range(x):
            """ Perturb Kepler magnitudes before calculating detection probability.
                Do this x times per star. Save 1 row per perturbed magnitude
                in save_xy() (each star has x rows). """

            if j == 0:
                kps = np.random.uniform(low=11, high=17, size=x)

            info['kic_kepmag'] = kps[j]


            # make the original Kepler PS
            if ds.sat == 'Kepler':
                ds.ts()
                ds.Periodogram()
                ds.kepler_noise(Kp=info['kic_kepmag'].as_matrix())
                ds.PS_add_noise()  # add noise to the power spectrum

            # transform the Kepler PS into TESS PS
            elif ds.sat == 'TESS':
                # units of exptime are seconds. noise in units of ppm
                ds.TESS_noise(imag=mag['Imag'].as_matrix(), exptime=30.*60.,\
                   teff=info['Teff'].as_matrix(), e_lat=mag['e_lat'].as_matrix(), sys_limit=0)
                ds.timeseries(plot_ts=False, plot_ps=False)

            # apply a detection test on every mode of the star
            star = DetTest(ds)
            star.get_snr(plt_PS=False, plt_SNR=False)  # calculate SNR values for every freq bin
            star.mode_snrs(v=False)  # SNR value at each mode
            star.Det_Prob(snrthresh=1.0, fap=0.05)  # detection probability value for each mode
            #star.Info2Save()
            # NOTE: plot results
            #star.Diagnostic_plot1()
            #star.Diagnostic_plot2()
            #star.Diagnostic_plot3()
            #star.plot4()



            #print list(vars(ds))
            #print star.prob


            # save the required X, Y data for Machine Learning
            output = data_for_ML(star)
            #output.fit_amplitudes()  # fit Gaussian to modes to get numax
            output.average_dnu()  # get dnu value from mode frequencies (only once per star)
            output.save_xy()

        #sys.exit()



    stop = timeit.default_timer()
    print(round(stop-start, 3), 'secs;', round((stop-start)/len(ts), 3), 's per star.')



#
