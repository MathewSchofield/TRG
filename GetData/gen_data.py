import K2data
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import glob
import sys
import os
warnings.simplefilter("ignore")

if __name__ == "__main__":
    df = pd.read_csv('20stars.csv')
    data_dir = '/home/davies/Dropbox/DATA_SETS/Kepler/APOKASC/Pins14/'

    star = K2data.Dataset(2001122017, '/home/davies/Data/ktwo_2001122017_llc.pow')
    star.power_spectrum()    

	"""
    print os.getcwd() + os.sep + '20Stars' + os.sep
    #sys.exit()
    for key, row in df.iterrows():
        kic = str(int(row.EPIC))
        print kic
        #sys.exit()
        files = glob.glob(os.getcwd() + os.sep + '20Stars' + os.sep + 'kplr*'+ '*.dat')
        print(kic)
        print(len(files))
        t = []
        y = []
        for f in files:
            print 'jhgf',f
            tmp = np.genfromtxt(f)
            print tmp
            sel = np.isfinite(tmp[:,3])
            tmp = tmp[sel,:]
            tmp = tmp[tmp[:,3] > 0, :]
            t = np.append(t, tmp[:,0])
            y = np.append(y, (tmp[:,3] / np.nanmean(tmp[:,3]) - 1.0) * 1e6) # now in ppm

        t = t[np.isfinite(y)]
        y = y[np.isfinite(y)]
        data = np.empty([len(t), 2])
        data[:,0] = t
        data[:,1] = y
        np.savetxt('Data/kplr' + kic + '_llc_concat.dat', data, \
                   header='# Time Flux')
	"""
