"""
Machine Learning using Random Forest and Multi Random Forest Regression.
Use Machine Learning to estimate the SNR values of stars.

X data: dnu, numax, magnitudes.
Y daya: SNR values.
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
from config import *  # the directories of the data files (ML_data_dir)

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

    def loadData(self):
        """ Load the X and Y data for the Kepler or TESS sample.
        Remove rows where all values are zero. Note: Kepler file does not have
        a 'Tobs' time in the filename. """

        if os.path.isfile(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv'):
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + str(self.Tobs) + '_XY.csv')
        else:
            self.xy = pd.read_csv(self.data_loc + '_' + self.sat + '_XY.csv')

        self.xy = self.xy.loc[(self.xy!=0).any(axis=1)]

    def pdet_bins(self, n=3, v=False):
        """ Assign discrete bins (i.e 0, 1, 2...) for the continuous Pdet values
        (i.e any value from 0.00 to 1.00), in order to apply Classification.
        n: the number of discrete bins to group pdet values into. """

        prob = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()
        if v:  print prob

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
            prob[(prob<=0.5)] = 0
            prob[(prob>0.5) & (prob<=0.9)] = 1
            prob[(prob>0.9) & (prob<1.0)] = 2

        elif n == 4:
            self.labels = [0, 1, 2, 3]  # to calculate precision
            prob[(prob<=0.4)] = 0
            prob[(prob>0.4) & (prob<=0.6)] = 1
            prob[(prob>0.6) & (prob<=0.9)] = 2
            prob[(prob>0.9) & (prob<=1.0)] = 3

        elif n == 5:
            self.labels = [0, 1, 2, 3, 4]  # to calculate precision
            prob[(prob<=0.2)] = 0
            prob[(prob>0.2) & (prob<=0.4)] = 1
            prob[(prob>0.4) & (prob<=0.6)] = 2
            prob[(prob>0.6) & (prob<=0.8)] = 3
            prob[(prob>0.8) & (prob<=1.0)] = 4

        self.xy[['Pdet1', 'Pdet2', 'Pdet3']] = prob
        self.n = n
        if v:  print prob

    def random_forest_classifier(self):
        """ Perform a Random Forest Classifier (made up of many decision trees)
        on the XY data. Y data must be given as discrete values
        e.g 0 or 1 for each mode (detected or not). """

        params = ['numax', 'Dnu', 'Teff', '[M/H]2', 'kic_kepmag', 'Bmag',
                  'Vmag', 'B-V', 'V-I', 'Imag']
        x = self.xy[params].as_matrix()
        y = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()

        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.3,
                                                            random_state=42)
        print self.sat, '; Tobs:', self.Tobs, ';', self.n, 'Y-data classes', '\n'
        # print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        # print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)
        #print 'y_test is a', (utils.multiclass.type_of_target(y_test))

        rfc = RandomForestClassifier(random_state=42, max_depth=100,
                                     max_features=10, min_samples_leaf=10)
        rfc = rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)  # predict on new data
        #print y_test, '\n', y_pred

        if self.n == 2:
            """ How well has the classifier done. Only works with binary labels """
            rfc_test = rfc.score(x_test, y_test)
            cv_score = cross_val_score(rfc, x_train, y_train)
            print 'DTC Test: ', rfc_test
            print "Accuracy:  %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2)
            print 'Accuracy:', accuracy_score(y_test, y_pred).mean(), accuracy_score(y_test, y_pred).std()
            print 'Classification report:', (classification_report(y_test, y_pred))
            print 'Confisuion matrix:', (confusion_matrix(y_test, y_pred))

        print 'Feature importance:', rfc.feature_importances_
        print 'Hamming loss:', np.sum(np.not_equal(y_test, y_pred))/float(y_test.size)

        av = 'weighted'
        print 'Precision1:', precision_score(y_test[:,0], y_pred[:,0], labels=self.labels, average=av)
        print 'Precision2:', precision_score(y_test[:,1], y_pred[:,1], labels=self.labels, average=av)
        print 'Precision3:', precision_score(y_test[:,2], y_pred[:,2], labels=self.labels, average=av)

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
        """ Make of a plot of the random_forest_regression() results. """

        plt.figure()
        s = 50
        a = 0.4
        plt.scatter(y_test[:, 0], y_test[:, 1], edgecolor='k',
                    c="navy", s=s, marker="s", alpha=a, label="Data")
        plt.scatter(y_multirf[:, 0], y_multirf[:, 1], edgecolor='k',
                    c="cornflowerblue", s=s, alpha=a,
                    label="Multi RF score=%.2f" % regr_multirf.score(x_test, y_test))
        plt.scatter(y_rf[:, 0], y_rf[:, 1], edgecolor='k',
                    c="c", s=s, marker="^", alpha=a,
                    label="RF score=%.2f" % regr_rf.score(x_test, y_test))
        plt.xlim([-6, 6])
        plt.ylim([-6, 6])
        plt.xlabel("target 1")
        plt.ylabel("target 2")
        plt.title("Comparing random forests and the multi-output meta estimator")
        plt.legend()
        plt.show()


if __name__ == '__main__':

    ml = Machine_Learning(data_loc=ML_data_dir, sat='Kepler', Tobs=365)
    ml.loadData()
    ml.pdet_bins()
    ml.random_forest_classifier()
    #ml.random_forest_regression()








#
