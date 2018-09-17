"""
Plot Templates for TRG
"""

import matplotlib.pyplot as plt


def generalPlot(title=None, xaxis=None, yaxis=None):

    plt.rc('font', size=14)
    gfig, ax = plt.subplots()

    if title != None:
        plt.title(title)
    if xaxis != None:
        plt.xlabel(xaxis)
    if yaxis != None:
        plt.ylabel(yaxis)

    return gfig, ax
