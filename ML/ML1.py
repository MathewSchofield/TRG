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

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split


class Machine_Learning(object):

    def __init__(self, data_loc):
        """  The location and filename of the X, Y data to load
        (without _X.csv or _Y.csv extension).  """
        self.data_loc = data_loc

    def loadData(self):
        """ Load the X and Y data. Remove rows where all values are zero. """
        self.x = pd.read_csv(self.data_loc + '_X.csv')
        self.y = pd.read_csv(self.data_loc + '_Y.csv')

        self.x = self.x.loc[(self.x!=0).any(axis=1)]
        self.y = self.y.loc[(self.y!=0).any(axis=1)]


    def random_forest_regression(self):
        """ Perform Random Forest Regression on the X, Y data.
            RF: Random Forest
            MRF: Multi Random Forest """

        #self.y = self.y[['SNR2', 'SNR3']]
        #print self.y
        #sys.exit()
        x = self.x[['Kp', 'dnu', 'numax']].as_matrix()
        y = self.y.as_matrix()

        test_size = 0.4  # use 30% of the data to test the algorithm (i.e 70% to train)
        random_state = 4  # ??
        max_depth = 4
        #random_state, max_depth = None, None

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

    ml = Machine_Learning(data_loc='/home/mxs191/Desktop/MathewSchofield/TRG/DetTest/DetTest1_results/data_for_ML/1000Stars/1000Stars')
    ml.loadData()
    ml.random_forest_regression()








#
