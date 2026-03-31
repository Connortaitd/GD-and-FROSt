# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:44:19 2020

@author: dylan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def load_mat_dah(path):

    data = loadmat(path)
    # The .mat files contains a matlab structure. This structure is imported
    # as a dictionary with keys:
    # trace
    # time
    # angfreq
    # delay
    # parameters
    # filtering
    # carrierAngFreq

    timeArray = np.array(data['M_trace']['time'][0][0].squeeze(), dtype=float) # time in fs
    omegaArray = np.array(data['M_trace']['angfreq'][0][0].squeeze(), dtype=float) # angular frequency in PHz
    delayArray = np.array(data['M_trace']['delay'][0][0].squeeze(), dtype=float) # delay of probe relative to switch
    trace = np.array(data['M_trace']['trace'][0][0].squeeze(), dtype=complex) # experimental data set
    parameters = np.array(data['M_trace']['parameters'][0][0].squeeze(), dtype=str) # Array size, time resolution and range, angular frequency resolution and range
    filtering = np.array(data['M_trace']['filtering'][0][0].squeeze(), dtype=str)
    carrierAngFreq = np.array(data['M_trace']['carrierAngFreq'][0][0].squeeze(), dtype=float)
    phase = np.array(data['M_trace']['phase'][0][0].squeeze(), dtype=float)


    return timeArray, omegaArray, delayArray, trace, carrierAngFreq, parameters, filtering, phase
