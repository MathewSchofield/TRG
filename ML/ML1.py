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
import matplotlib.gridspec as gridspec

TRG = os.getcwd().split('ML')[0]
sys.path.insert(0, TRG)
from plotTemplates import generalPlot
from config import *  # the directories of the data files (ML_data_dir and pins_floc)

sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset

from sklearn import preprocessing, utils
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, confusion_matrix,\
    accuracy_score, precision_score


class Machine_Learning(object):

    def __init__(self, data_loc, sat, Tobs, plx_source):
        """  The file location of the X, Y data to load, the satellite to test
        ('Kepler' or 'TESS') and the observing timeself.
        Set plx_source to be 'dr2' or 'tgas'. """
        self.data_loc = data_loc
        self.sat = sat
        self.Tobs = Tobs
        self.plx_source = plx_source

    def get_parallaxes(self):
        """ Get parallaxes for the 1000 stars from 'tgas' or 'dr2'. """

        if self.plx_source == 'dr2':
            plx = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars_simbad_DR2.csv', sep=';', skipinitialspace=True)
            plx.dropna(inplace=True)
            plx['KIC'] = plx['identifier        ']
            plx['parallax'] = plx['plx  ']
            plx['KIC'] = plx['KIC'].str.strip()
            plx['parallax'] = plx['parallax'].str.strip()
            plx[['KIC', 'parallax']].to_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars_simbad_DR2_2.csv', index=False)

        if self.plx_source == 'tgas':
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

    def loadData(self, v=False, add_logg=True, add_parallax=True, clas=True):
        """ 1.  Load the X and Y data for the Kepler or TESS sample (Note: Kepler
                file does not have a 'Tobs' time in the filename.)
            2.  Remove rows where all values are zero.
            3.  add_logg (Bool; kewarg):      Add log(g) values from Pinsonneualt (2014).
            4.  add_parallax (Bool; kewarg):  Add parallax values from TGAS or DR2
                                              (see get_parallaxes() function).
            5.  clas (Bool; kewarg):          Load stellar classifications file
                                              from Elsworth (2016).  """

        if os.path.isfile(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv'):
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv')
        else:
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + '_XY.csv')

        if v:  print self.xy.shape

        self.xy = self.xy.loc[(self.xy!=0).any(axis=1)]
        if v:  print self.xy.shape

        if clas:
            evo = pd.read_csv(clas_floc, sep='\s+')
            #evo['KIC_number'] = 'KIC ' + evo['KIC_number']

            # print self.xy.shape
            self.xy = pd.merge(left=self.xy, right=evo,
                left_on='KIC', right_on='KIC_number', how='inner')
            # print self.xy.shape, list(self.xy)
            #print self.xy[['KIC', 'KIC_number', 'Classification']]
        if v:  print self.xy.shape

        if add_logg:
            pins = pd.read_csv(pins_floc, sep=';')
            self.xy = pd.merge(left=self.xy, right=pins[['KIC', 'log.g1', 'log.g2']],
                            left_on='KIC', right_on='KIC', how='inner')
            self.xy = self.xy[self.xy['log.g1'] != '         ']  # remove rows without log(g) values
            self.xy[['log.g1', 'log.g2']] = self.xy[['log.g1', 'log.g2']].apply(pd.to_numeric, errors='coerce')
        if v:  print self.xy.shape

        if add_parallax:

            self.xy['KIC'] = 'KIC ' + self.xy['KIC'].astype(str).str[:-2].str.strip().str.rstrip()

            if self.plx_source == 'dr2':
                plx = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars_simbad_DR2_2.csv')
                plx = plx[plx['parallax']!='~']
                self.xy = pd.merge(left=self.xy, right=plx[['KIC', 'parallax']], left_on='KIC', right_on='KIC', how='inner')

            if self.plx_source == 'tgas':
                plx = pd.read_csv(plx_floc)
                self.xy = pd.merge(left=self.xy, right=plx[['kic', 'tyc', 'parallax']], left_on='KIC', right_on='kic', how='inner')
        if v:  print self.xy.shape

    def pdet_bins(self, n=3, v=False, plot=False):
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

    def perturb(self):
        """ Perturb the Kp and Imag values to increase the sample size (unused). """
        x = 1  # number of iterations to loop through every star (number of different magnitudes per star)
        pdf_range = [10.2, 12., 100]  # range of Kp magnitudes for the PDF

        # rather than using a uniform distribution in magnitude, use a PDF of
        # the Kepler noise function to get the a distribution of magnitudes
        ds = Dataset([], [], sat='Kepler', bandpass=1, Tobs=27)  # Tobs in days
        rand_mags = ds.rvs_from_noise_function(pdf_range = pdf_range, x=x)
        #ds.Plot2()  # plot the noise functions

        for j in range(x):
            """ Perturb the Kepler/TESS magnitudes 'x' times before calculating
                detection probability. Do this x times per star. Save 1 row per
                perturbed magnitude in save_xy() (each star has x rows). """
            pass
            diff = rand_mags[j]-float(mag['Imag'])  # change magnitudes for this iteration
            mag[['Imag', 'Vmag', 'Bmag']] += diff
            info['kic_kepmag'] += diff
        sys.exit()

    def random_forest_classifier(self, subset='all', save=False):
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
        print self.sat, '; Tobs:', self.Tobs, ';', 'subset:', subset, self.n, 'Y-data classes', '\n'

        rfc = RandomForestClassifier(random_state=42, max_depth=100,
                                     min_samples_leaf=5)#, max_features=5)
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

        print 'Hamming loss:', hl
        print 'Precision', (p1+p2+p3)*100/3.
        # print 'Feature importance:', rfc.feature_importances_
        # print 'Precision1:', p1
        # print 'Precision2:', p2
        # print 'Precision3:', p3

        self.rfc = rfc  # to use in Plot1()
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

    def random_forest_regression(self, test=True):
        """ Perform Random Forest Regression on the X data (Teff, [M/H], Kp),
            Y data (Imag) to calculate Imag for missing stars.
            RF: Random Forest
            MRF: Multi Random Forest

            Kewargs
            test (bool) True:  Test the algorithm to make sure it is robust.
                        False: Use the algorithm to calculate Imags for missing stars.
            """

        allstars = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars.csv')
        training = pd.read_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars_simbad4.csv')

        allstars['KIC'] = 'KIC ' + allstars['KIC'].astype(str)
        training['KIC'] = training['KIC'].str.strip().str.rstrip()
        self.data = pd.merge(left=allstars[['KIC', 'numax', 'dnu', 'Teff', '[M/H]2', 'kic_kepmag']],
            right=training[['KIC', 'Imag', 'e_lng', 'e_lat']], left_on='KIC', right_on='KIC', how='left')


        if test == True:
            """ Only use stars that have Kp and Imag values for the training
            and testing sets. """
            subset = (self.data['Imag']==self.data['Imag'])
        elif test == False:
            """ The training set are all stars with Kp and Imag values.
            The testing set are the stars without Imag values. """
            subset = (self.data==self.data)

        x = self.data[['Teff', '[M/H]2', 'kic_kepmag']][subset].as_matrix()
        y = self.data[['Imag']][subset].as_matrix()


        if test == True:
            x_train, x_test, y_train, y_test = train_test_split(x, y,
                test_size=0.8, random_state=42)
        elif test == False:
            x_train = x[(self.data['Imag']==self.data['Imag']).as_matrix()]
            y_train = y[(self.data['Imag']==self.data['Imag']).as_matrix()]
            x_test  = x[(self.data['Imag']!=self.data['Imag']).as_matrix()]
            y_test  = y[(self.data['Imag']!=self.data['Imag']).as_matrix()]


        print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)


        # 1. make an instance of the RF algorithm called 'regr_rf'
        # 2. train it on the training self.dataset
        # 3. make predcitions about new y self.data
        regr_rf = RandomForestRegressor(random_state=42, max_depth=3,
            n_estimators=4, min_weight_fraction_leaf=0.01)
        regr_rf.fit(x_train, y_train)  # create the RF algorithm
        y_rf = regr_rf.predict(x_test)  # predict on new self.data with RF
        if test == True:
            rf_test = regr_rf.score(x_test, y_test)
            print 'RF Test: ', rf_test
            self.y_test = y_test

        print 'mean:', np.mean(self.y_test[:,0]-y_rf)
        print 'sd:', np.std(self.y_test[:,0]-y_rf)

        self.y_rf = y_rf
        self.y_train = y_train
        self.test = test


        if test == False:
            """ Save the predicted Imags for the stars without them. """
            #print 'dont save'; sys.exit()
            self.data['Imag'][(self.data['Imag']!=self.data['Imag'])] = y_rf
            self.data['KIC'] = self.data['KIC'].apply(lambda x: x.split(' ')[1])
            self.data.to_csv('/home/mxs191/Desktop/MathewSchofield/TRG/GetData/1000Stars/1000stars_2.csv', index=False)

        # 1. make an instance of the MRF algorithm called 'regr_multirf'
        # 2. train it on the training dataset
        # 3. make predcitions about new y data
        # regr_multirf = MultiOutputRegressor(RandomForestRegressor(random_state=42))
        # regr_multirf.fit(x_train, y_train)  # create the MRF algorithm
        # y_multirf = regr_multirf.predict(x_test)  # predict on new data with MRF
        # multirf_test = regr_multirf.score(x_test, y_test)  # how well has MRF done?
        # print 'MRF Test:', rf_test

    def Plot1(self):
        """ Make of a plot of the Feature Importance.
        http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html """

        importances = self.rfc.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.rfc.estimators_],
                     axis=0)
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking in order
        print("Feature ranking:")
        for f in range(len(importances)):
            print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

        fig = plt.figure()
        plt.rc('font', size=14)
        plt.bar(range(len(importances)), importances[indices],
               color="lightblue", yerr=std[indices], align="center")
        plt.xticks(range(len(importances)), [r'$I_{\rm mag}$',r'$\rm log(g)$',r'$T_{\rm eff}$',r'$\pi$','[M/H]'])
        plt.ylabel("Feature importance")
        plt.xlim([-1, len(importances)])
        plt.show()
        fig.savefig('Plot1_featureimportance.pdf')

    def Plot2(self):
        """ Find a relationship between Kp and Imag, so that measured Imags
        are not needed. This will increase the number of datasets available. """

        plt.rc('font', size=14)
        fig, ax = plt.subplots()
        plt.hist(self.xy['kic_kepmag']-self.xy['Imag'], bins=50, color="lightblue")
        plt.xlabel(r'$K_{p} - I_{\rm mag}$')
        plt.ylabel('Number of Stars')
        plt.show()
        fig.savefig('Plot2_KpImag.pdf')

    def Plot3(self, plot='Imag'):
        """ Plot a histogram to compare the predicted Imags from
        random_forest_regression() with the known values. """

        plt.rc('font', size=14)
        fig, ax = plt.subplots()

        if plot == 'Imag':
            """ Compare the Imags used to train RFR, with the Imags predicted for the stars without them by RFR """

            if self.test == True:
                step = 0.1
                widths = np.arange(min(self.y_train), max(self.y_train)+step, step)
                plt.hist(self.y_test, label='Test', bins=widths, alpha=0.5)
                plt.hist(self.y_rf, label='Predicted', histtype='step',color="k", bins=widths)
                plt.xlabel(r'$I_{\rm mag}$ / mag')
                plot = 'Imag_tested'

            if self.test == False:
                step = 0.1
                widths = np.arange(min(self.y_train), max(self.y_train)+step, step)
                plt.hist(self.y_train, label='Trained', bins=widths, alpha=0.5)
                plt.hist(self.y_rf, label='Predicted', histtype='step', color="k", bins=widths)
                plt.xlabel(r'$I_{\rm mag}$ / mag')
                plot = 'Imag_trained'

        elif plot == 'Kp':
            step = 0.05
            widths = np.arange(min(self.data['kic_kepmag']), max(self.data['kic_kepmag'])+step, step)
            plt.hist(self.data['kic_kepmag'][self.data['Imag']==self.data['Imag']], label='Trained', bins=widths, alpha=0.5)
            plt.hist(self.data['kic_kepmag'][self.data['Imag']!=self.data['Imag']], label='Predicted', color="k", histtype='step', bins=widths)
            plt.xlabel(r'$K_{p}$ / mag')

        plt.ylabel('Number of Stars')
        plt.legend()
        plt.show()
        fig.savefig('Plot3_%s_distribution.pdf' % plot)

    def Plot4(self):
        """ Make a scatter plot of the difference between true and predicted
        Imag values for the test sample. """

        fig = plt.figure()#figsize=(14, 16))
        plt.rc('font', size=14)
        G = gridspec.GridSpec(2, 2, width_ratios=(4,1))
        line = np.linspace(8, 13, 100)

        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax1.scatter(self.y_test[:,0], self.y_rf)
        ax1.plot(line, line, c='k')
        ax1.set_ylabel(r'Pred. $I_{\rm mag}$ / mag')

        ax2 = plt.subplot(G[1, 0])
        ax2.scatter(self.y_test[:,0], self.y_test[:,0]-self.y_rf)
        ax2.plot(line, np.zeros(100), c='k')
        ax2.set_xlabel(r'True $I_{\rm mag}$ / mag')
        ax2.set_ylabel(r'True-Pred. $I_{\rm mag}$ / mag')

        ax3 = plt.subplot(G[1, 1])
        import seaborn as sns
        sns.kdeplot(self.y_test[:,0]-self.y_rf, shade=True, vertical=True, \
                    ax=ax3, bw=0.4)
        plt.show()
        fig.savefig('Plot4_Imag_scatter.pdf')


if __name__ == '__main__':

    ml = Machine_Learning(data_loc=ML_data_dir, sat='TESS', Tobs=27,
                          plx_source='dr2')
    #ml.get_parallaxes()
    #ml.loadData()
    #ml.pdet_bins()
    #ml.random_forest_classifier()
    ml.random_forest_regression()
    #ml.Plot1()
    #ml.Plot2()
    #ml.Plot3()
    ml.Plot4()





#
