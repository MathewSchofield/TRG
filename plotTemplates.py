"""
Plot Templates for TRG
"""

import matplotlib.pyplot as plt


def generalPlot(title=None, xaxis=None, yaxis=None):

    plt.rc('font', size=24)
    gfig, ax = plt.subplots(figsize=(12.0, 14.0))

    if title != None:
        plt.title(title)
    if xaxis != None:
        plt.xlabel(xaxis)
    if yaxis != None:
        plt.ylabel(yaxis)

    return gfig, ax
