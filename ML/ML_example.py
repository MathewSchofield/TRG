"""
A commented Machine Example.

This is just example code from
http://scikit-learn.org/stable/auto_examples/ensemble/plot_random_forest_regression_multioutput.html#sphx-glr-auto-examples-ensemble-plot-random-forest-regression-multioutput-py
"""

import warnings
warnings.simplefilter(action = "ignore")
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

#import sklearn
#from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

TRG = os.getcwd().split('ML')[0]
sys.path.insert(0, TRG)
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG + 'likeTESS' + os.sep)
from likeTESS1 import getInput


if __name__ == '__main__':
    """ Machine Learning using Random Forest and Multi Random Forest Regression.
    RF: Random Forest
    MRF: Multi Random Forest
    """

    # make some random data
    rng = np.random.RandomState(1)
    x = np.sort(200 * rng.rand(1000, 1) - 100, axis=0)
    y = np.array([np.pi * np.sin(x).ravel(), np.pi * np.cos(x).ravel()]).T
    y += (0.5 - rng.rand(*y.shape))
    print 'x & y shape', np.shape(x), np.shape(y)


    test_size = 0.3  # use 30% of the data to test the algorithm (i.e 70% to train)
    random_state = 10  # ??
    max_depth = 30

    x_train, x_test, y_train, y_test = train_test_split(x,
                                                        y,
                                                        test_size=test_size,
                                                        random_state=random_state)
    print 'train', len(x_train), len(y_train)
    print 'test', len(x_test), len(y_test)


    # 1. make an instance of the RF algorithm called 'regr_rf'
    # 2. train it on the training dataset
    # 3. make predcitions about new y data
    regr_rf = RandomForestRegressor(max_depth=max_depth, random_state=random_state)
    regr_rf.fit(x_train, y_train)  # create the RF algorithm
    y_rf = regr_rf.predict(x_test)  # predict on new data with RF
    rf_score = regr_rf.score(x_test, y_test)
    print 'RF score', rf_score

    # 1. make an instance of the MRF algorithm called 'regr_multirf'
    # 2. train it on the training dataset
    # 3. make predcitions about new y data
    regr_multirf = MultiOutputRegressor(RandomForestRegressor(max_depth=max_depth,
                                                             random_state=random_state))
    regr_multirf.fit(x_train, y_train)  # create the MRF algorithm
    y_multirf = regr_multirf.predict(x_test)  # predict on new data with MRF
    mrf_score = regr_multirf.score(x_test, y_test)
    print 'MRF score', mrf_score

    """
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
    """










#
