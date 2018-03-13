"""
Machine Learning using a Random Forest Classifier.
Use Machine Learning to estimate the Pdet values of stars.

X data: dnu, numax, magnitudes.
Y daya: Pdet values.
"""

import warnings
warnings.simplefilter(action = "ignore")
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

TRG = os.getcwd().split('ML')[0]
sys.path.insert(0, TRG)
from plotTemplates import generalPlot
from config import *  # the directories of the data files (ML_data_dir and pins_floc)

from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix,\
    accuracy_score, precision_score


class Machine_Learning(object):

    def __init__(self, data_loc, sat, Tobs):
        """  The file location of the X, Y data to load, the satellite to test
        ('Kepler' or 'TESS') and the observing time """
        self.data_loc = data_loc
        self.sat = sat
        self.Tobs = Tobs

    def get_parallaxes(self):
        """ Use Tycho2 IDs from Simbad to make a query to TGAS,
        to download and match parallaxes with TYC and KIC IDs.
        Match this file in loadData()"""

        # NOTE: Step 1: Go from Simbad file with Kic and TYC ID file ready for TGAS with only TYC
        a = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_KICTYC2.tsv', sep='\s+')
        b = pd.DataFrame(a['typed'].str.split('   ',1).tolist())  # get the Tycho ID values only
        b = 'TYC ' + b[0]  # add 'TYC ' onto the start of the ID name
        c = b[b.str.contains('-')]  # only keep stars that have TYC IDs to search in TGAS
        print b.shape, c.shape
        print c
        c.to_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_TYC.csv', index=False)
        sys.exit()


        # NOTE: Step 2: Get every TGAS TYC ID and parallax in 1 file to match with the TYC IDs of the Kepler Red Giants
        for i in range(16):  # loop through the files
            if i<10:
                i = '0' + str(i)
            i = str(i)
            print i

            tgas = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/TGAS/tgas_parts/TgasSource_000-000-0' + i + '.csv')
            print tgas.shape

            if i == '00':
                plx = tgas[['tycho2_id', 'parallax']]
            else:
                plx = pd.concat([plx, tgas[['tycho2_id', 'parallax']]])

        print plx.shape
        plx.dropna(inplace=True)
        print plx.shape
        plx['tycho2_id'] = 'TYC ' + plx['tycho2_id']
        plx.to_csv('/home/mxs191/Desktop/phd_y2/Gaia/TGAS/TYC_plx.csv', index=False)
        sys.exit()


        # NOTE: Step 3: Merge the TYC values of the 1000stars to the TYC and plx values from the entire DR1
        tyc  = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_TYC.csv', names=['TYC'])
        tgas = pd.read_csv('/home/mxs191/Desktop/phd_y2/Gaia/TGAS/TYC_plx.csv')
        print tyc
        print tgas
        print tyc.shape, tgas.shape, list(tyc), list(tgas)
        both = pd.merge(left=tyc, right=tgas[['tycho2_id', 'parallax']], left_on='TYC', right_on='tycho2_id', how='inner')
        print both
        both[['tycho2_id', 'parallax']].to_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/IDs/TYC_plx.csv', index=False)
        sys.exit()


        # NOTE: Step 4: clean up dataframe with KIC and TYC IDs. Merge with parallaxes
        plx = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/IDs/TYC_plx.csv')

        ids = pd.read_csv('/home/mathew/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_KICTYC.csv', sep=';|')
        ids.rename(columns={'typed ident ':'kic', '                    identifier':'tyc'}, inplace=True)
        ids['tyc'] = ids['tyc'].astype(str).str[:-12].str.strip()
        ids['kic'] = ids['kic'].astype(str).str.strip().str.rstrip()
        ids.to_csv('/home/mathew/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_KICTYC2.csv', index=False)

        both = pd.merge(left=ids, right=plx, left_on='tyc', right_on='tycho2_id', how='inner')
        both.to_csv('/home/mathew/Desktop/MathewSchofield/TRG/GetData/IDs/1000stars_KICTYC_plx.csv', index=False)
        sys.exit()

    def loadData(self, add_logg=True, add_parallax=True, clas=True):
        """ 1.  Load the X and Y data for the Kepler or TESS sample (Note: Kepler
                file does not have a 'Tobs' time in the filename.)
            2.  Remove rows where all values are zero.
            3.  add_logg (Bool; kewarg):      Add log(g) values from Pinsonneualt (2014).
            4.  add_parallax (Bool; kewarg):  Add parallax values from TGAS
                                              (see get_parallaxes() function).
            5. clas (Bool; kewarg):            Load stellar classifications file
                                              from Elsworth (2016).  """

        if os.path.isfile(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv'):
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv')
        else:
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + '_XY.csv')

        self.xy = self.xy.loc[(self.xy!=0).any(axis=1)]

        if clas:
            evo = pd.read_csv(clas_floc, sep='\s+')
            #evo['KIC_number'] = 'KIC ' + evo['KIC_number']

            # print self.xy.shape
            self.xy = pd.merge(left=self.xy, right=evo,
                left_on='KIC', right_on='KIC_number', how='inner')
            # print self.xy.shape, list(self.xy)
            #print self.xy[['KIC', 'KIC_number', 'Classification']]

        if add_logg:
            pins = pd.read_csv(pins_floc, sep=';')
            self.xy = pd.merge(left=self.xy, right=pins[['KIC', 'log.g1', 'log.g2']],
                            left_on='KIC', right_on='KIC', how='inner')
            self.xy = self.xy[self.xy['log.g1'] != '         ']  # remove rows without log(g) values
            self.xy[['log.g1', 'log.g2']] = self.xy[['log.g1', 'log.g2']].apply(pd.to_numeric, errors='coerce')

        if add_parallax:
            plx = pd.read_csv(plx_floc)

            self.xy['KIC'] = 'KIC ' + self.xy['KIC'].astype(str).str[:-2].str.strip().str.rstrip()
            self.xy = pd.merge(left=self.xy, right=plx[['kic', 'tyc', 'parallax']], left_on='KIC', right_on='kic', how='inner')

    def pdet_bins(self, n=3, v=True, plot=False):
        """ Assign discrete bins (i.e 0, 1, 2...) for the continuous Pdet values
        (i.e any value from 0.00 to 1.00), in order to apply Classification.
        n: the number of discrete bins to group pdet values into. """

        prob = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()
        if v:  print prob
        if plot:
            plt.hist(self.xy['Pdet1'], bins=100)
            plt.hist(self.xy['Pdet2'], bins=100)
            plt.hist(self.xy['Pdet3'], bins=100)
            plt.show()
            sys.exit()

        if n == 2:
            self.labels = [0, 1]  # to calculate precision
            if self.Tobs == 27:
                prob[(prob<=0.5)] = 0
                prob[(prob>0.5)]  = 1
            else:
                prob[(prob<=0.9)] = 0
                prob[(prob>0.9)]  = 1

        elif n == 3:
            self.labels = [0, 1, 2]  # to calculate precision
            prob[(prob>0.9) & (prob<1.0)] = 2
            prob[(prob>0.5) & (prob<=0.9)] = 1
            prob[(prob<=0.5)] = 0

        elif n == 4:
            self.labels = [0, 1, 2, 3]  # to calculate precision
            prob[(prob>0.7) & (prob<=1.0)] = 3
            prob[(prob>0.5) & (prob<=0.7)] = 2
            prob[(prob>0.3) & (prob<=0.5)] = 1
            prob[(prob<=0.3)] = 0

        elif n == 5:
            self.labels = [0, 1, 2, 3, 4]  # to calculate precision
            prob[(prob>0.8) & (prob<=1.0)] = 4
            prob[(prob>0.6) & (prob<=0.8)] = 3
            prob[(prob>0.4) & (prob<=0.6)] = 2
            prob[(prob>0.2) & (prob<=0.4)] = 1
            prob[(prob<=0.2)] = 0

        elif n == 6:
            self.labels = [0, 1, 2, 3, 4, 5]  # to calculate precision
            prob[(prob>0.9) & (prob<=1.0)] = 5
            prob[(prob>0.8) & (prob<=0.9)] = 4
            prob[(prob>0.6) & (prob<=0.8)] = 3
            prob[(prob>0.4) & (prob<=0.6)] = 2
            prob[(prob>0.2) & (prob<=0.4)] = 1
            prob[(prob<=0.2)] = 0

        self.xy[['Pdet1', 'Pdet2', 'Pdet3']] = prob
        self.n = n
        if v:  print prob
        if v:  print '% rows in \'0\' class:', len(prob[prob==0])/float(prob.shape[0]*prob.shape[1])
        if v:  print '% rows in \'1\' class:', len(prob[prob==1])/float(prob.shape[0]*prob.shape[1])
        if v:  print '% rows in \'2\' class:', len(prob[prob==2])/float(prob.shape[0]*prob.shape[1])
        if v:  print '% rows in \'3\' class:', len(prob[prob==3])/float(prob.shape[0]*prob.shape[1])
        if v:  print '% rows in \'4\' class:', len(prob[prob==4])/float(prob.shape[0]*prob.shape[1])

    def random_forest_classifier(self, subset='2CL', save=True):
        """ Perform a Random Forest Classifier (made up of many decision trees)
        on the XY data. Y data must be given as discrete values
        e.g 0 or 1 for each mode (detected or not).

        subset ('all', 'RGB', 'RC', '2CL'):  If not 'all' stars, only use a
                                             subset of the stars.
        save (Bool; kewarg):  Save the precision and hamming loss of
                              the results to a file.  """

        if subset!='all':
            self.xy = self.xy[self.xy['Classification']==subset]
            print self.xy.shape

        # x_labels = ['Teff', '[M/H]2', 'kic_kepmag', 'Bmag',
        #             'Vmag', 'B-V', 'V-I', 'Imag', 'log.g1', 'parallax']
        x_labels = ['Teff', '[M/H]2', 'kic_kepmag', 'log.g1', 'parallax']
        x = self.xy[x_labels].as_matrix()
        y = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.3,
                                                            random_state=42)
        print self.sat, '; Tobs:', self.Tobs, ';', self.n, 'Y-data classes', '\n'

        rfc = RandomForestClassifier(random_state=42, max_depth=100,
                                     min_samples_leaf=10)#, max_features=8)
        rfc = rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)  # predict on new data

        p1 = precision_score(y_test[:,0], y_pred[:,0], labels=self.labels, average='weighted')
        p2 = precision_score(y_test[:,1], y_pred[:,1], labels=self.labels, average='weighted')
        p3 = precision_score(y_test[:,2], y_pred[:,2], labels=self.labels, average='weighted')
        hl = np.sum(np.not_equal(y_test, y_pred))/float(y_test.size)

        if self.n == 2:
            """ How well has the classifier done. Only works with 2 (binary) labels """
            rfc_test = rfc.score(x_test, y_test)
            cv_score = cross_val_score(rfc, x_train, y_train)
            print 'DTC Test: ', rfc_test
            print "Accuracy:  %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2)
            print 'Accuracy:', accuracy_score(y_test, y_pred).mean(), accuracy_score(y_test, y_pred).std()
            print 'Classification report:', (classification_report(y_test, y_pred))

        print 'Feature importance:', rfc.feature_importances_
        print 'Hamming loss:', hl
        print 'Precision', (p1+p2+p3)*100/3.
        # print 'Precision1:', p1
        # print 'Precision2:', p2
        # print 'Precision3:', p3

        if save:
            if self.sat == 'Kepler': self.Tobs = 4*365  # days

            if os.path.isfile('ML1_results.csv'):  # add to the file
                output = pd.read_csv('ML1_results.csv')
                #print output

                row_to_add = {'Sat':self.sat, 'Tobs':self.Tobs, 'Subset':subset,
                     'Number_of_stars':len(self.xy), 'Pres1':p1, 'Pres2':p2,
                     'Pres3':p3, 'HL':hl}
                row_to_add = pd.DataFrame(data=row_to_add, index=[0])
                output = output.append(row_to_add, ignore_index=True)
                output.drop_duplicates(subset=['Sat', 'Tobs', 'Subset'], inplace=True)

                output = output[['Sat','Tobs','Subset','Number_of_stars','Pres1',
                    'Pres2','Pres3','HL']]  # save the file in this order
                output.to_csv('ML1_results.csv', index=False)
                #print output

            else:
                d = {'Sat':self.sat, 'Tobs':self.Tobs, 'Subset':subset,
                     'Number_of_stars':len(self.xy), 'Pres1':p1, 'Pres2':p2,
                     'Pres3':p3, 'HL':hl}
                output = pd.DataFrame(data=d, index=[0])
                output = output[['Sat','Tobs','Subset','Number_of_stars','Pres1',
                    'Pres2','Pres3','HL']]  # save the file in this order
                output.to_csv('ML1_results.csv', index=False)  # make the file

    def random_forest_regression(self):
        """ Perform Random Forest Regression on the X, Y data.
            RF: Random Forest
            MRF: Multi Random Forest """

        params = ['numax', 'Dnu', 'Teff', '[M/H]2', 'kic_kepmag', 'Bmag',
                  'Vmag', 'B-V', 'V-I', 'Imag']
        x = self.xy[params].as_matrix()
        y = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()

        test_size = 0.2  # use 30% of the data to test the algorithm (i.e 70% to train)
        random_state = 42  # keep this constant to keep the results constant
        max_depth = 4

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=test_size,
                                                            random_state=random_state)

        print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)


        # 1. make an instance of the RF algorithm called 'regr_rf'
        # 2. train it on the training dataset
        # 3. make predcitions about new y data
        regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
        regr_rf.fit(x_train, y_train)  # create the RF algorithm
        y_rf = regr_rf.predict(x_test)  # predict on new data with RF
        rf_test = regr_rf.score(x_test, y_test)  # how well has RF done:
        print 'RF Test: ', rf_test

        # 1. make an instance of the MRF algorithm called 'regr_multirf'
        # 2. train it on the training dataset
        # 3. make predcitions about new y data
        regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                                 random_state=random_state))
        regr_multirf.fit(x_train, y_train)  # create the MRF algorithm
        y_multirf = regr_multirf.predict(x_test)  # predict on new data with MRF
        multirf_test = regr_multirf.score(x_test, y_test)  # how well has MRF done?
        print 'MRF Test:', rf_test

    def Plot1(self):
        """ Make of a plot of the random_forest_classifier() results. """


        plt.hist()

if __name__ == '__main__':

    ml = Machine_Learning(data_loc=ML_data_dir, sat='TESS', Tobs=27)
    #ml.get_parallaxes()
    ml.loadData()
    ml.pdet_bins()
    ml.random_forest_classifier()









#
