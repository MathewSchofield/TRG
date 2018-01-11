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

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class Machine_Learning(object):

    def __init__(self, data_loc):
        """  The location and filename of the X, Y data to load
        (without _X.csv or _Y.csv extension).  """
        self.data_loc = data_loc

    def loadData(self):
        """ Load the X and Y data. Remove rows where all values are zero. """

        self.xy = pd.read_csv(self.data_loc + '_XY.csv')
        self.xy = self.xy.loc[(self.xy!=0).any(axis=1)]

    def decision_tree_classifier(self):
        """ Perform a Decision Tree Classifier on the XY data. """

        rs = 42  # random state

        x = self.xy[['KIC', 'numax', 'Dnu', 'Teff', '[M/H]2', 'kic_kepmag',
                      'Bmag', 'Vmag', 'B-V', 'V-I', 'Imag']].as_matrix()
        y = self.xy[['Pdet1', 'Pdet2', 'Pdet3']].as_matrix()
        y = np.mean(y, axis=1)

        print (y*100).astype(int)
        y = (y*100).astype(int)



        x_train, x_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.5,
                                                            random_state=rs)

        print 'x training/testing set: ', np.shape(x_train), '/', np.shape(x_test)
        print 'y training/testing set: ', np.shape(y_train), '/', np.shape(y_test)

        #print y_train

        # x_train = [[0, 0], [1, 1]]
        # y_train = [0., 1.]
        # x_test  = [[2,2]]
        # y_test  = [1.]
        # print np.shape(x_train), '/', np.shape(x_test)
        # print np.shape(y_train), '/', np.shape(y_test)

        # from sklearn import preprocessing
        # from sklearn import utils
        # lab_enc = preprocessing.LabelEncoder()
        # encoded = lab_enc.fit_transform(y_test)
        # print(utils.multiclass.type_of_target(y_test))
        # print(utils.multiclass.type_of_target(y_test.astype('int')))
        # print(utils.multiclass.type_of_target(encoded))
        # sys.exit()

        dtc = DecisionTreeClassifier(random_state=rs)
        dtc = dtc.fit(x_train, y_train)
        y_predict = dtc.predict(x_test)  # predict on new data
        dtc_test = dtc.score(x_test, y_test)  # how well has the classifier done
        print 'DTC Test: ', dtc_test
        print y_predict
        print y_test
        print accuracy_score(y_test, y_predict)
        print dtc.scores_


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

    ml = Machine_Learning(data_loc=ML_data_dir)
    ml.loadData()
    ml.decision_tree_classifier()
    #ml.random_forest_regression()








#
