import numpy as np
import pandas as pd
import scipy.ndimage as ndim
import sys
import glob
import gatspy
from gatspy.periodic import LombScargleFast
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from astropy.stats import mad_std
from astropy.convolution import Gaussian1DKernel, convolve
import matplotlib.gridspec as gridspec


"""
References:
(1)  Scargle, J, D. (1982) 'Studies in astronomical time series analysis. II -
     Statistical aspects of spectral analysis of unevenly spaced data'
(2)  Jake Vanderplas 2017 'Understanding the Lomb-Scargle Periodogram'
(3)  Kp magnitude conversion: https://keplerscience.arc.nasa.gov/the-kepler-space-telescope.html
(4)  'The connection between stellar granulation and oscillation as seen by the Kepler mission'
	 by Kallinger et al (2014)
(5)  'Predicting the detectability of oscillations in solar-type stars observed by Kepler'
	 by Chaplin (2011)
"""


class Dataset(object):
    def __init__(self, epic, data_file, bandpass, Tobs):
        ''' Initial contruct for the Dataset object

        Parameters
        ----------------
        epic: int
            The epic number for the source

        data_file: str
            The path to the file containg the data

        bandpass: Float
            The bandpass of the observations

        Tobs: int (days)
            The observation time of the observations

        '''

        self.epic = epic
        self.data_file = data_file
        self.Tobs = Tobs  # ovservation length (days)
        self.time = []  # the un-adjusted time
        self.flux = []  # the un-adjusted flux
        self.time_fix = []  # the adjusted time
        self.flux_fix = []  # the adjusted flux
        self.freq = []
        self.power = []
        self.smoo_freq = []
        self.smoo_power = []
        self.bin_width = []  # mu Hz
        self.KPnoise = []    # Kepler instrumental noise (ppm)
        self.TESSnoise = []  # TESS instrumental noise (ppm)
        self.numax = []
        self.bandpass = bandpass
        self.Pgran = []
        self.vnyq = 1e6*(2*30.*60.)**-1  # mu Hz

    def get_modes(self, id_file):

        self.id_file = id_file  # mode ID file loc
        self.mode_id = pd.read_csv(self.id_file)

    def ts(self, verbose=False, sigma_clip=4):
        '''  Reads in a timeseries from the file stored in data file ONLY.
        All data modification is to be done in the frequency domain.
        '''

        data = np.genfromtxt(self.data_file)
        data[:,1] = data[np.argsort(data[:,0]),1]  # re-order flux by time
        data[:,0] = data[np.argsort(data[:,0]),0]  # re-order time

        self.time = (data[:,0] - data[0,0]) * 24.0 * 3600.0  # start time at 0 secs
        self.flux = data[:,1]

        self.flux = self.flux[np.argsort(self.time)]
        self.time = self.time[np.argsort(self.time)]

        # remove data gaps
        self.time = self.time[(self.flux != 0) & (np.isfinite(self.flux))]
        self.flux = self.flux[(self.flux != 0) & (np.isfinite(self.flux))]

        self.flux = self.flux[np.argsort(self.time)]
        self.time = self.time[np.argsort(self.time)]

        self.flux_fix = self.flux
        sel = np.where(np.abs(self.flux_fix) < mad_std(self.flux_fix) * sigma_clip)
        self.flux_fix = self.flux_fix[sel]  # remove extreme values
        self.time_fix = self.time[sel]      # remove extreme values

        if verbose:
            print("Read file {}".format(self.data_file))
            print("Data points : {}".format(len(self.time)))

    def Periodogram(self, madVar=True):
        """ This function computes the power spectrum from the timeseries ONLY.
        """

        dtav = np.mean(np.diff(self.time_fix))  # mean value of time differences (s)
        dtmed = np.median(np.diff(self.time_fix))  # median value of time differences (s)
        if dtmed == 0:  dtmed = dtav

        # compute periodogram from regular frequency values
        fmin = 0  # minimum frequency
        N = len(self.time_fix)  # n-points
        df = 1./(dtmed*N)  # bin width (1/Tobs) (in Hz)
        model = LombScargleFast().fit(self.time_fix, self.flux_fix, np.ones(N))
        power = model.score_frequency_grid(fmin, df, N/2)  # signal-to-noise ratio, (1) eqn 9
        freqs = fmin + df * np.arange(N/2)  # the periodogram was computed over these freqs (Hz)

        # the variance of the flux
        if madVar:  var = mad_std(self.flux_fix)**2
        else:       var = np.std(self.flux_fix)**2

        # convert to PSD, see (1) eqn 1, 8, 9 & (2)
        power /= np.sum(power)  # make the power sum to unity (dimensionless)
        power *= var  # Parseval's theorem. time-series units: ppm. variance units: ppm^2
        power /= df * 1e6  # convert from ppm^2 to ppm^2 muHz^-1

        if len(freqs) < len(power):  power = power[0:len(freqs)]
        if len(freqs) > len(power):  freqs = freqs[0:len(power)]

        self.freq = freqs * 1e6    # muHz
        self.power = power         # ppm^2 muHz^-1
        self.bin_width = df * 1e6  # muHz

    def timeseries(self, sigma_clip=4, plot_ts=False, plot_ps=False):
        '''  Reads in a timeseries from the file stored in data file.
        This works for ascii files that can be read by np.genfromtxt. The
        function assumes that time is in the zero column and flux is in the
        one column.

        The data is read in, zero values are removed, and stored in the time and flux.

        A sigma clip is performed on the flux to remove extreme values.  The
        level of the sigma clip can be adjusted with the sigma_clip parameter.
        The results of the sigma clip are stored in time_fix and flux_fix.


        Parameters
        ------------------
        sigma_clip: Float
            The level at which to perform the sigma clip.  If sigma_clip=0
            then no sigma is performed.

        noise: Float
            If noise is not zero then additional noise is added to the timeseries where the value of noise is the standard deviation of the additional noise.

        '''

        self.ts(sigma_clip=4)

        """ find the 27 days in the timeseries (inc. gaps) with the minimum time gaps """
        self.time = self.time / 86400             # convert from seconds to days
        self.time_fix = self.time_fix / 86400     # convert from seconds to days
        diffs = np.full(len(self.time_fix), np.inf)

        # the data gaps over the reduced length of data
        for i, item in enumerate(self.time_fix):

            if len(self.time_fix[i:i+self.Tobs]) == self.Tobs:  # don't go beyond the end of the data
                diffs[i] = np.sum(np.diff(self.time_fix[i:i+self.Tobs]))

        mn = np.where(diffs==np.min(diffs))[0][0]  # the data with minimum gap lengths
        strt = abs(self.Tobs - self.time_fix[mn:mn+self.Tobs*48][-1]) # the start time to get 'days' length dataset
        idx = (np.abs(self.time_fix-strt)).argmin()  # index of adjusted start time

        self.time_fix = self.time_fix[idx:mn+self.Tobs*48]  # cut the data down
        self.flux_fix = self.flux_fix[idx:mn+self.Tobs*48]  # cut the data down

        self.time = self.time * 86400.            # convert from days to seconds
        self.time_fix = self.time_fix * 86400.    # convert from days to seconds

        if self.bandpass != 1.:  # adjust the bandpass
            self.flux_fix = self.flux_fix * self.bandpass

        if self.TESSnoise > 0.0:  # add noise in the time domain
            self.flux_fix += np.random.randn(len(self.time_fix)) * self.TESSnoise

        self.Periodogram()  # make the power spectrum

        if plot_ts:  self.plot_timeseries()
        if plot_ps: self.plot_power_spectrum()

    def power_spectrum(self, verbose=False, madVar=True, Plot1=False,\
        plot_ts=False, plot_ps=False):
        ''' This function computes the power spectrum from the timeseries.

        The function checks to see if the timeseries has been read in, and if not it calls the read_timeseries function.

        The properties of the power spectrum can be altered for a given timeseries via the noise, and length parameters.

        The frequency and power are stored in the object atributes self.freq and self.power.

        Parameters
        ----------------
        verbose: Bool(False)
            Provide verbose output if set to True.

        madVar: Bool
            the Median Absolute Deviation (robust variance of input data)

        plot_ps: Bool
            plots the power spectrum at different stages of analysis from Kepler to TESS

        Returns
        ----------------

        NA

        Examples
        ----------------

        To read in a data set and create the power spectrum one need only run:

        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
        >>> star.power_spectrum()

        '''
        # read the timeseries data WITHOUT adding noise in the time domain
        # add noise and change the length of the dataset in the frequency domain
        if len(self.time) < 1:  self.ts()

        self.Periodogram()  # compute periodogram. freqs (Hz) power (ppm^2 mu Hz^-1)

        if self.bandpass != 1.:  # adjust the bandpass
            self.power *= self.bandpass**2

        # Smooth the data using a convolve function to remove chi^2 2 DOF noise
        self.Smoo(smoo_scale=5)

        #self.KPnoise = np.mean(self.power[-100:]) # estimate of Kepler noise level at high frequencies
        self.power -= self.KPnoise  # subtract Kepler noise
        if self.TESSnoise > 0.0:  self.power += self.TESSnoise  # add TESS noise to get 'limit' spectrum

        # reduce the data set length
        tbw = 1e6/(self.Tobs*86400)  # binwidth of reduced dataset (mu Hz; self.Tobs in days)
        tfreqs = np.arange(0., self.freq[-1], tbw)  # frequency bins for TESS dataset, mu Hz
        tpower = np.interp(tfreqs, self.freq, self.power)  # power values at these freqs

        if Plot1: self.Plot1(tfreqs, tpower)  # plot power spectrum at different stages

        # add chi^2 2 DOF noise
        s = np.random.uniform(low=0.0, high=1.0, size=len(tpower))
        tpower = -tpower * np.log(s)

        self.freq = tfreqs        # mu Hz
        self.power = tpower       # ppm^2 mu Hz^-1
        self.bin_width = tbw      # Hz

        if plot_ts: self.plot_timeseries()
        if plot_ps: self.plot_power_spectrum()

        if verbose:
            print("Frequency resolution : {}".format(self.freq[1]))
            print("Nyquist : ~".format(self.freq.max()))

    def TESS_noise(self, imag, exptime, teff, e_lat = 30, g_lng = 96, g_lat = -30, subexptime = 2.0,\
    frac_aper = 0.76, e_pix_ro = 10, geom_area = 60.0, pix_scale = 21.1, sys_limit = 0):

        N = np.ceil(10.0**-5.0 * 10.0**(0.4*(20.0-imag)))
        npix_aper = 10*(N+10)

        omega_pix = pix_scale**2.0
        n_exposures = exptime/subexptime

        # electrons from the star
        megaph_s_cm2_0mag = 1.6301336 + 0.14733937*(teff-5000.0)/5000.0
        e_star = 10.0**(-0.4*imag) * 10.0**6 * megaph_s_cm2_0mag * geom_area * exptime * frac_aper
        e_star_sub = e_star*subexptime/exptime

        # e/pix from zodi
        dlat = (abs(e_lat)-90.0)/90.0
        vmag_zodi = 23.345 - (1.148*dlat**2.0)
        e_pix_zodi = 10.0**(-0.4*(vmag_zodi-22.8)) * (2.39*10.0**-3) * geom_area * omega_pix * exptime

        # e/pix from background stars
        dlat = abs(g_lat)/40.0*10.0**0

        dlon = g_lng
        q = np.where(dlon>180.0)
        if len(q[0])>0:
        	dlon[q] = 360.0-dlon[q]

        dlon = abs(dlon)/180.0*10.0**0
        p = [18.97338*10.0**0, 8.833*10.0**0, 4.007*10.0**0, 0.805*10.0**0]
        imag_bgstars = p[0] + p[1]*dlat + p[2]*dlon**(p[3])
        e_pix_bgstars = 10.0**(-0.4*imag_bgstars) * 1.7*10.0**6 * geom_area * omega_pix * exptime

        # compute noise sources
        noise_star = np.sqrt(e_star) / e_star
        noise_sky  = np.sqrt(npix_aper*(e_pix_zodi + e_pix_bgstars)) / e_star
        noise_ro   = np.sqrt(npix_aper*n_exposures)*e_pix_ro / e_star
        noise_sys  = 0.0*noise_star + sys_limit/(1*10.0**6)/np.sqrt(exptime/3600.0)

        noise1 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0)
        noise2 = np.sqrt(noise_star**2.0 + noise_sky**2.0 + noise_ro**2.0 + noise_sys**2.0)

        self.TESSnoise = noise2 * 1e6  # in ppm

    def kepler_noise(self, Kp):
        """ calculate the noise for a source from Kepler """
        c = 1.28 * 10**(0.4*(12.-Kp) + 7.)  # detections per cadence, (5) eqn 17.
        self.KPnoise = 1e6/c * np.sqrt(c + 9.5 * 1e5*(14./Kp)**5) # in ppm

    def granulation(self, numax, dilution=1):
        """ Estimate the power due to granulation from (4) """

        self.numax = numax
    	a_nomass = self.bandpass * 3382*self.numax**-0.609
    	b1 = 0.317 * self.numax**0.970
    	b2 = 0.948 * self.numax**0.992

        # Divide by dilution squared as it affects stars in the time series.
        # The units of dilution change from ppm to ppm^2 microHz^-1 when going from the
        # time series to frequency. p6: c=4 and zeta = 2*sqrt(2)/pi
        Pgran = (((2*np.sqrt(2))/np.pi) * (a_nomass**2/b1) / (1 + ((self.freq/b1)**4)) \
        + ((2*np.sqrt(2))/np.pi) * (a_nomass**2/b2) / (1 + ((self.freq/b2)**4))) / (dilution**2)

        # From (9). the amplitude suppression factor. Normalised sinc with pi (area=1)
        eta = np.sinc((self.freq/(2*self.vnyq)))

        # the granulation after attenuation
        self.Pgran = Pgran * eta**2

    def plot_power_spectrum(self, smoo=0, plog=True):
        ''' Plots the power spectrum '''
        if len(self.freq) < 0:
            self.power_spectrum()

        fig, ax = plt.subplots()
        ax.plot(self.freq, self.power, 'b-')
        if plog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(r'Frequency ($\rm \mu Hz$)')
        ax.set_ylabel(r'Power ($\rm ppm^{2} \, \mu Hz^{-1}$)')
        ax.set_xlim([self.freq.min(),self.freq.max()])
        ax.set_title('KIC ' + str(self.epic))

        if smoo > 0:
            self.rebin_quick(smoo)
            ax.plot(self.smoo_freq, self.smoo_power, 'k-', linewidth=4)

        plt.show()
        fig.savefig('power_spectrum_' + str(self.epic) + '.png')

    def plot_timeseries(self):
        ''' Plots the time series '''
        if len(self.time) < 0:
            self.read_data()
        fig, ax = plt.subplots()
        ax.plot(self.time, self.flux, 'b.')
        ax.plot(self.time_fix, self.flux_fix, 'k.')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Flux (ppm)')
        ax.set_title('KIC ' + str(self.epic))
        fig.savefig('timeseries_' + str(self.epic) + '.png')

    def Plot1(self, tfreqs, tpower):
        """ Plot the power spectrum at different stages from converting """
        plt.plot(self.freq, self.power, label='TESS noise level')
        plt.plot(self.freq, self.power + self.KPnoise - self.TESSnoise, 'r--',
            alpha=0.4, label='Kepler noise level')
        plt.plot(tfreqs, tpower, label='TESS noise level and binwidth')
        plt.legend(loc='lower left')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r'$\nu / \mu$Hz')
        plt.ylabel(r'Power $\rm ppm^{2} \mu Hz^{-1}$')
        plt.show()

    def Smoo(self, smoo_scale):
        """ smooth the input power using a convolve function """

        g = Gaussian1DKernel(stddev=smoo_scale)
        self.power = convolve(self.power, g, boundary='extend')

    def Diagnostic_plot1(self, scale):
        """ subplots of TESS-like spectra made in the time and frequency
        domains, observed over different timescales """

        fig = plt.figure(figsize=(12, 16))
        plt.rc('font', size=26)
        smoo_scale = 10  # scale to convolve over

        # axes positions
        ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1)
        ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=1)
        ax3 = plt.subplot2grid((3, 2), (1, 1), rowspan=1)
        ax4 = plt.subplot2grid((3, 2), (2, 0))
        ax5 = plt.subplot2grid((3, 2), (2, 1))

        # axes properties
        for idx, i in enumerate([ax1, ax2, ax3, ax4, ax5]):
            i.set_yscale('log')
            i.set_xscale('log')
            titles = [r'$Kepler \ (4 \rm \ yrs)$',\
                r'TESS (1 yr, time domain)', r'TESS (27 days, time domain)',\
                r'TESS (1 yr, freq domain)', r'TESS (27 days, freq domain)']
            i.set_title(titles[idx])

            if (i == ax4) or (i == ax5):
                i.set_xlabel(r'$\rm \nu / \mu Hz$')

            if (i == ax1) or (i == ax2) or (i == ax4):
                i.set_ylabel(r'$\rm PSD / ppm^{2} \mu Hz^{-1}$')

            # set the axes scale. 'full': the entire spectrum; 'modes': just modes
            if scale == 'full':
                i.set_ylim([0.02,367665])
                i.set_xlim([1.5, 299])

            if scale == 'modes':
                i.set_ylim([40,143796])
                i.set_xlim([9, 117])


        # original Kepler spectrum. smooth with 1 muHz moving mean
        self.ts()
        self.Periodogram()
        ax1.plot(self.freq, self.power, color='k', alpha=0.5)
        self.Smoo(smoo_scale)
        ax1.plot(self.freq, self.power, color='k')
        #smoo = ndim.filters.uniform_filter1d(self.power, size=int(1./self.bin_width))
        #ax1.plot(self.freq, smoo, color='r')

        # TESS-like, conversion in the time series, 1 year
        self.Tobs = 365  # days
        self.timeseries()
        ax2.plot(self.freq, self.power, color='c')
        self.Smoo(smoo_scale)
        ax2.plot(self.freq, self.power, color='k')

        # TESS-like, conversion in the time series, 27 days
        self.Tobs = 27  # days
        self.power_spectrum()
        ax3.plot(self.freq, self.power, color='c')
        self.Smoo(smoo_scale)
        ax3.plot(self.freq, self.power, color='k')

        # TESS-like, conversion in the freq domain, 1 year
        self.Tobs = 365  # days
        self.power_spectrum()
        ax4.plot(self.freq, self.power, color='b', alpha=0.8)
        self.Smoo(smoo_scale)
        ax4.plot(self.freq, self.power, color='k')

        # TESS-like, conversion in the freq domain, 27 days
        self.Tobs = 27  # days
        self.power_spectrum()
        ax5.plot(self.freq, self.power, color='b', alpha=0.8)
        self.Smoo(smoo_scale)
        ax5.plot(self.freq, self.power, color='k')

        plt.tight_layout()
        plt.show()
        fig.savefig('diagnostic_plot1_' + scale + '.pdf')

    def Diagnostic_plot2(self, scale):
        """ overplot smoothed TESS-like spectra made in the time and frequency
        domains, observed over different timescales """

        fig = plt.figure(figsize=(12, 16))
        plt.rc('font', size=26)
        smoo_scale = 10  # scale to convolve over
        plt.xscale('log')
        plt.yscale('log')
        plt.ylabel(r'$\rm PSD / ppm^{2} \mu Hz^{-1}$')
        plt.xlabel(r'$\rm \nu / \mu Hz$')

        if scale == 'full':
            plt.ylim([10,124916])
            plt.xlim([1.5, 299])

        if scale == 'modes':
            plt.ylim([40,143796])
            plt.xlim([9, 117])

        # original Kepler spectrum. smooth with 1 muHz moving mean
        self.ts()
        self.Periodogram()
        self.Smoo(smoo_scale)
        plt.plot(self.freq, self.power, color='gray', alpha=0.4, label=r'$Kepler \ (4 \rm \ yrs)$')
        #smoo = ndim.filters.uniform_filter1d(self.power, size=int(1./self.bin_width))
        #ax1.plot(self.freq, smoo, color='r')

        # TESS-like, conversion in the time series, 1 year
        self.Tobs = 365  # days
        self.timeseries()
        self.Smoo(smoo_scale)
        plt.plot(self.freq, self.power, color='r', alpha=0.4, label=r'TESS (1 yr, time domain)')

        # TESS-like, conversion in the time series, 27 days
        self.Tobs = 27  # days
        self.power_spectrum()
        self.Smoo(smoo_scale)
        plt.plot(self.freq, self.power, color='b', alpha=0.4, label=r'TESS (27 days, time domain)')

        # TESS-like, conversion in the freq domain, 1 year
        self.Tobs = 365  # days
        self.power_spectrum()
        self.Smoo(smoo_scale)
        plt.plot(self.freq, self.power, color='g', alpha=0.4, label=r'TESS (1 yr, freq domain)')

        # TESS-like, conversion in the freq domain, 27 days
        self.Tobs = 27  # days
        self.power_spectrum()
        self.Smoo(smoo_scale)
        plt.plot(self.freq, self.power, color='orange', alpha=0.6, label=r'TESS (27 days, freq domain)')

        plt.tight_layout()
        plt.legend(loc='bottom left')
        plt.show()
        fig.savefig('diagnostic_plot2_' + scale + '.pdf')

    def Diagnostic(self, Kp, imag, exptime, teff, e_lat):
        """ Perform diagnostic tests to check TESS power spectra """

        self.TESS_noise(imag, exptime, teff, e_lat, sys_limit=0)
        self.kepler_noise(Kp)
        #self.Diagnostic_plot1(scale='full')
        #self.Diagnostic_plot1(scale='modes')
        #self.Diagnostic_plot2(scale='full')
        self.Diagnostic_plot2(scale='modes')

#
