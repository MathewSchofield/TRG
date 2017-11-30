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


if __name__ == '__main__':
    print 'k'
