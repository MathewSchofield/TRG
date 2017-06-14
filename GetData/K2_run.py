#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 13:30:12 2017

@author: davies
"""

import os
#import sys
import warnings
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy import interpolate
import pandas as pd
#import seaborn as sns
import yaml
import itertools

from K2data import Dataset
from K2psps import PSPS
from K2sps import SPS
from K2ts import TS
from K2psacf import PSACF
from K2tsacf import TSACF
from K2wavelet import  WAVELET
from K2ps_flicker import  PS_FLICKER

class ConfigurationNotFound(Exception):
    """Raised if the config.yml file could not be found."""

def readConfiguration():

    """
    Get our YAML settings.
    """

    try:
        conf_dir = os.environ['K2PATH']
    except KeyError:
        conf_dir = '.'

    sFilespec = conf_dir + '/config.yml'

    try:
        with open(sFilespec, 'r') as fd:
            settings = yaml.safe_load(fd)
    except FileNotFoundError as exc:
        raise ConfigurationNotFound('Could not open {}'.format(sFilespec)) from exc

    return settings

def main():
    ''' Make a run from the configuration file '''
    settings = readConfiguration()

    if settings['ignore_warnings']:
        warnings.filterwarnings('ignore')

    run = settings['run']
    pipeline = settings[run]
    data_dir = settings['work_dir'] + pipeline['data_dir']

    pprint = False
    stars = pd.read_csv(data_dir + pipeline['csv_file'])
    epics = stars[pipeline['star_id']].tolist()

    for idx, epic in enumerate(epics):
        data_file = data_dir + 'kplr'+str(epic)+'_llc_concat.dat'
        ds = Dataset(epic, data_file)
        ds.power_spectrum(length=48*80*pipeline['cams'], noise=0)
        #pipes = [SPS(ds), PSPS(ds), TS(ds), PSACF(ds), TSACF(ds), WAVELET(ds), PS_FLICKER(ds)]
        pipes = [PS_FLICKER(ds)]
        try:
            res = [n() for n in pipes]
            if idx == 0:
                cols = ['EPIC']
                for r in res:
                    cols += r.Name.tolist()
                    cols += [n + '_err' for n in r.Name.tolist()]
                df = pd.DataFrame(columns=cols)
            vals = [epic]
            for r in res:
                vals += r.Value.tolist()
                vals += r.Err.tolist()
            df.loc[len(df)] = vals
        except:
            print("Failed on ", epic)
    df.to_csv(data_dir + pipeline['output'], index=False)

if __name__ == "__main__":
    main()
