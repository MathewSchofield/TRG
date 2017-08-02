"""
Use KDE to estimate the probability of detection for a star, given some input
parameters (e.g Tobs, dnu and numax)
"""

import warnings
warnings.simplefilter(action = "ignore")
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

TRG = os.getcwd().split('Parameters2Pdet')[0]
sys.path.insert(0, TRG)
from plotTemplates import generalPlot
sys.path.insert(0, TRG + 'GetData' + os.sep)
from K2data import Dataset
sys.path.insert(0, TRG + 'likeTESS' + os.sep)
from likeTESS1 import getInput


if __name__ == '__main__':
    print 'k'
