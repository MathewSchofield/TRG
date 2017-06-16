import numpy as np
import sys
import gatspy
from gatspy.periodic import LombScargleFast
import matplotlib.pyplot as plt
from astropy.stats import mad_std
from astropy.convolution import Gaussian1DKernel, convolve
from scipy.interpolate import interp1d

"""
References:
(1)  Scargle, J, D. (1982) 'Studies in astronomical time series analysis. II -
     Statistical aspects of spectral analysis of unevenly spaced data'
(2)  Jake Vanderplas 2017 'Understanding the Lomb-Scargle Periodogram'
"""


class Dataset(object):
    def __init__(self, epic, data_file):
        ''' Initial contruct for the Dataset object

        Parameters
        ----------------
        epic: int
            The epic number for the source

        data_file: str
            The path to the file containg the data

        Returns
        -----------------
        NA

        Examples
        -----------------

        This is just creating the object, so for epic='2001122017' and
        data file of '/home/davies/Data/ktwo_2001122017_llc.pow' you would run:

        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')

        '''

        self.epic = epic
        self.data_file = data_file
        self.time = []  # the un-adjusted time
        self.flux = []  # the un-adjusted flux
        self.time_fix = []  # the adjusted time
        self.flux_fix = []  # the adjusted flux
        self.freq = []
        self.power = []
        self.smoo_freq = []
        self.smoo_power = []
        self.bin_width = []

    def read_timeseries(self, verbose=False, sigma_clip=4, start=0, length=-1,\
        bandpass=1., noise=0):
        '''  Reads in a timeseries from the file stored in data file.
        This works for ascii files that can be read by np.genfromtxt. The
        function assumes that time is in the zero column and flux is in the
        one column.

        Time should be in units of days.

        The data is read in, zero values are removed, and stored in the time and flux.

        A sigma clip is performed on the flux to remove extreme values.  The
        level of the sigma clip can be adjusted with the sigma_clip parameter.
        The results of the sigma clip are stored in time_fix and flux_fix.


        Parameters
        ------------------
        verbose: Bool(False)
            Set to true to produce verbose output.

        sigma_clip: Float
            The level at which to perform the sigma clip.  If sigma_clip=0
            then no sigma is performed.

        start: Int
            The # points at which to start the selection of points

        length: Int
            The # points to have in the selection of points.

        bandpass: Float
            Adjusts the bandpass of the flux

        noise: Float
            If noise is not zero then additional noise is added to the timeseries where the value of noise is the standard deviation of the additional noise.

        Returns
        ------------------
        NA

        Examples
        ------------------

        To load in the time series with a 4 sigma clip, one would run:

        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
        >>> star.read_timeseries()

        '''
        data = np.genfromtxt(self.data_file)
        self.time = (data[:,0] - data[0,0])# * 24.0 * 3600.0  # start time at 0 secs
        self.flux = data[:,1]

        self.flux = self.flux[np.argsort(self.time)]
        self.time = self.time[np.argsort(self.time)]

        #print len(self.time)
        #sys.exit()

        # remove data gaps
        self.time = self.time[np.isfinite(self.flux)]
        self.flux = self.flux[np.isfinite(self.flux)]
        self.time = self.time[self.flux != 0]
        self.flux = self.flux[self.flux != 0]

        """ find the 27 days in the timeseries with the minimum time gaps """
        print len(self.time)
        print np.diff(self.time)
        print self.time
        print np.min(np.sum(np.diff(self.time)))

        print length
        for i, item in enumerate(self.time):
            print i, item
            print self.time[i:i+27]
            sys.exit()

        #plt.plot(self.time[0:-1], np.diff(self.time))
        #plt.plot(np.diff(self.time))
        #plt.show()
        sys.exit()



        self.flux = self.flux[np.argsort(self.time)]
        self.time = self.time[np.argsort(self.time)]

        self.flux_fix = self.flux
        sel = np.where(np.abs(self.flux_fix) < mad_std(self.flux_fix) * sigma_clip)
        self.flux_fix = self.flux_fix[sel]  # remove extreme values
        self.time_fix = self.time[sel]      # remove extreme values

        if bandpass != 1.:  # adjust the bandpass
            self.flux_fix = self.flux_fix * bandpass

        if verbose:
            print("Read file {}".format(self.data_file))
            print("Data points : {}".format(len(self.time)))

        a = start
        if length == -1:
            b = len(self.time_fix)
        else:
            b = start + length
            if b > len(self.time_fix):
                b = len(self.time_fix)
                a = len(self.time_fix) - length
        self.sel = range(a, b, 1)  # the range to cut the data down to

        self.time_fix = self.time_fix[self.sel]  # cut the data down
        self.flux_fix = self.flux_fix[self.sel]  # cut the data down

        if noise > 0.0:  # add noise in the time domain
            self.flux_fix += np.random.randn(len(self.time_fix)) * noise


    def read_psd(self):
        ''' This function reads in a power spectrum for self.data_file.  This
        module currently supports .txt, .pow, and .fits. The frequencies and
        power are stored in the self.freq and self.power object properties.

        .txt files must have frequency in the zero column and power in the one
        column.  Frequency is expected to be in units of Hz and will be stored
        as muHz.

        .pow files must have frequency in the zero column and power in the one column.  Frequency is expected to be in units of muHz and will be stored as muHz.

        .fits files must have frequency in the zero column and power in the one column of the data object.  Frequency is expected to be in units of Hz and will be stored as muHz.

        Parameters
        ----------------

        NA

        Returns
        ----------------
        NA

        Examples
        ----------------

        To read in a power spectrum:

        >>> import K2data
        >>> star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc_psd_v1.pow')
        >>> star.read_psd()

        '''
        if self.data_file.endswith('.txt'):
            data = np.genfromtxt(self.data_file)
            self.freq = data[:,0]*1e6
            self.power = data[:,1]
        elif self.data_file.endswith('.pow'):
            data = np.genfromtxt(self.data_file)
            self.freq = data[:,0]
            self.power = data[:,1]
        elif self.data_file.endswith('.fits'):
            import pyfits
            data = pyfits.getdata(self.data_file)
            data = np.array(data)
            self.freq = data[:,0]*1e6
            self.power = data[:,1]
        else:
            print("File type not supported!")

    def power_spectrum(self, verbose=False, noise=0.0, \
                       length=-1, start=0, madVar=False):
        ''' This function computes the power spectrum from the timeseries.

        The function checks to see if the timeseries has been read in, and if not it calls the read_timeseries function.

        The properties of the power spectrum can be altered for a given timeseries via the noise, and length parameters.

        The frequency and power are stored in the object atributes self.freq and self.power.

        Parameters
        ----------------
        verbose: Bool(False)
            Provide verbose output if set to True.

        noise: Float
            If noise is not zero then additional noise is added to the timeseries where the value of noise is the standard deviation of the additional noise.

        length: Int
            If length is not -1 then a subset of the timeseries is selected when n points will equal length.

        start: Int
            The points to use on the transform will be taken from [start:start+length]

        madVar: Bool
            the Median Absolute Deviation (robust variance of input data)

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
        # add noise and change the length of the dataset in the frequency doamin
        if len(self.time) < 1:
            self.read_timeseries(verbose=False, start=start, \
                bandpass=0.85, noise=0, length=-1)

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
        power /= np.sum(power)  # make the power sum to unity
        power *= var  # periodogram gives SNR. so multiply by var get power due to signal (ppm)
        power /= df * 1e6  # convert to ppm^2 muHz^-1

        if len(freqs) < len(power):  power = power[0:len(freqs)]
        if len(freqs) > len(power):  freqs = freqs[0:len(power)]

        #plt.plot(freqs*1e6, power)

        # Smooth the data using a convolve function to remove chi^2 2 DOF noise
        g = Gaussian1DKernel(stddev=5)
        power = convolve(power, g, boundary='extend')

        kp_noise = np.mean(power[-100:]) # estimate of Kepler noise level at high frequencies
        power -= kp_noise  # subtract Kepler noise
        if noise > 0.0:  power += noise  # add TESS noise to get 'limit' spectrum

        # reduce the data set length
        tbw = 1./(length*86400)  # binwidth of reduced dataset (Hz; obs. length in days)
        tfreqs = np.arange(0., freqs[-1], tbw)  # frequency bins for TESS dataset
        tpower = np.interp(tfreqs, freqs, power)  # power values at these freqs

        #plt.plot(freqs*1e6, power)
        #plt.plot(freqs, power + kp_noise - noise, 'r--', alpha=0.4)
        #plt.plot(tfreqs*1e6, tpower)
        #plt.xscale('log')
        #plt.yscale('log')
        #plt.show()
        #sys.exit()

        # add chi^2 2 DOF noise
        s = np.random.uniform(low=0.0, high=1.0, size=len(tpower))
        tpower = -tpower * np.log(s)

        self.freq = tfreqs * 1e6  # mu Hz
        self.power = tpower
        self.bin_width = tbw  # bin width (Hz)

        if verbose:
            print("Frequency resolution : {}".format(self.freq[1]))
            print("Nyquist : ~".format(self.freq.max()))

    def pixel_cost(self, x):
        """ returns the number of pixels in the TESS aperture
        """

        N = np.ceil(10.0**-5.0 * 10.0**(0.4*(20.0-x)))
        N_tot = 10*(N+10)

        return N_tot

    def calc_noise(self, imag, exptime, teff, e_lng = 0, e_lat = 30, g_lng = 96, g_lat = -30, subexptime = 2.0, npix_aper = 10, \
    frac_aper = 0.76, e_pix_ro = 10, geom_area = 60.0, pix_scale = 21.1, sys_limit = 0):

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

        return noise2

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

    def rebin_quick(self, smoo):
        ''' TODO Write DOC strings '''
        if smoo < 1:
            return f, p
        if self.freq == []:
            self.power_spectrum()
        self.smoo = int(smoo)
        m = int(len(self.power) / self.smoo)
        self.smoo_freq = self.freq[:m*self.smoo].reshape((m,self.smoo)).mean(1)
        self.smoo_power = self.power[:m*self.smoo].reshape((m,self.smoo)).mean(1)
