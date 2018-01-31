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
from config import *  # the directories to find the data files (ML_data_dir)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import accuracy_score
from sklearn import preprocessing, utils


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

    def random_forest_classifier(self):
        """ Perform a Random Forest Classifier (made up of many decision trees)
        on the XY data. Y data must be given as 0 or 1 for each mode (detected or not). """

        rs = 42  # random state

        params = ['numax', 'Dnu', 'Teff', '[M/H]2', 'kic_kepmag', 'Bmag',
                  'Vmag', 'B-V', 'V-I', 'Imag']
        x = self.xy[params].as_matrix()
        y = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()

        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.3,
                                                            random_state=rs)
        print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)
        #print 'y_test is a', (utils.multiclass.type_of_target(y_test))

        rfc = RandomForestClassifier(random_state=rs, max_depth=100, max_features=10,
            min_samples_leaf=10)
        rfc = rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)  # predict on new data
        rfc_test = rfc.score(x_test, y_test)  # how well has the classifier done
        print 'DTC Test: ', rfc_test
        print 'Feature importance:', rfc.feature_importances_

        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        print(classification_report(y_test, y_pred))
        #print(confusion_matrix(y_test, y_pred))
        #cv_score = cross_val_score(rfc, x_train, y_train)
        #print("Accuracy: %0.2f (+/- %0.2f)" % (cv_score.mean(), cv_score.std() * 2))

    def random_forest_regression(self):
        """ Perform Random Forest Regression on the X, Y data.
            RF: Random Forest
            MRF: Multi Random Forest """

        #self.y = self.y[['SNR2', 'SNR3']]
        #print self.y
        #sys.exit()
        x = self.x[['Kp', 'dnu', 'numax']].as_matrix()
        y = self.y.as_matrix()

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

    ml = Machine_Learning(data_loc=ML_data_dir, sat='TESS', Tobs=27)
    ml.loadData()
    ml.random_forest_classifier()
    #ml.random_forest_regression()








#
