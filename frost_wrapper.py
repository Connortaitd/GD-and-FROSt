# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:36:33 2020

@author: dylan
"""

#%%
''' Import functions and modules '''

import sys
sys.path.insert(0, 'C:/Users/Connor Davis/Documents/Research/FROSt/code/FROST retrieval - ptychography/')
sys.path.insert(1, 'C:/Users/dylan/Documents/GitHub/FROSt/code/FROST retrieval - ptychography/')
sys.path.insert(2, 'C:/Users/D/Documents/myGitHub/FROSt/code/FROST retrieval - ptychography/')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from scipy import interpolate
import scipy.fftpack
from scipy import integrate
import scipy.io as sio
import frost_functions as ff
from load_mat import load_mat
from load_mat_dah import load_mat_dah
import pickle



#%% Updated code 10/26/2021

useAbsSwitch = 1

# path = 'C:/Users/dylan/Documents/myGitHub/FROSt/data/'
# path = 'C:/Users/dylan/Documents/GitHub/FROG-PCGPA/data/'
# path = 'C:/Users/D/Documents/myGitHub/FROSt/data/'
path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/'

# folder = '20211024/Ando Scans/'
# filename = 'processed trace.mat' # 200 delay FWHM
# filename = 'processed trace higher res.mat' # 200 delay FWHM
# filename = 'processed trace higher res 75delayFWHM.mat'
# filename = 'processed trace 3res 2^10.mat' # 200 delay FWHM For Ando scans

# folder = '20211026/1 pass scans/'
# folder = '20211026/2 pass scans/'
# filename = 'processed trace.mat'

# folder = '20220214 FROSt measurement/1 pass scans/'
folder = '20260227 FROSt/'
filename = 'processed trace.mat' # 200 delay FWHM For Ando scans

totalPath = path + folder + filename

timeArray, omegaArray, delayArray, data, carrierAngFreq, _, _ = load_mat_dah(totalPath)
data = data.T

numFFT = len(omegaArray)
numDelayArray = len(delayArray)
dt = 2 * np.pi / (omegaArray[-1] - omegaArray[0])
omegaArrayCenter = 0 # Carrier angular frequency was already subtracted in processing.
omegaArrayShifted = omegaArray - omegaArrayCenter

## Generate switchTimeArray
numSwitch = int(round((max(delayArray) - min(delayArray))/dt) + numFFT + 1)
# numSwitch = numFFT # Length of switch using rolling method
print(numSwitch)

switchTimeArrayShift = (abs(max(delayArray)) - abs(min(delayArray)))/2 # Shift to correct for delay arrays that aren't symmetric about 0
switchTimeArray = np.arange(-numSwitch//2, numSwitch//2) * dt + switchTimeArrayShift
print(len(switchTimeArray))

omegaArrayRetrieval = np.fft.ifftshift(omegaArrayShifted)
dataRetrieval = np.fft.ifftshift(data, axes=1) # ifftshift intensity data so shifts do not need to be performed throughout code

len(omegaArray)


#%%
''' Preview data '''

# sns.set_context('poster')

# fig, ax = plt.subplots(figsize = (8,6))
# im = ax.pcolormesh( omegaArray, delayArray, data )
# im = ax.pcolormesh( omegaArray0, delayArray0, data0 )
# fig.colorbar(im)
# plt.title('Measured Data')
# plt.title('I($\omega$, $\\tau$)')
# ax.set_xlabel('$\omega$ (PHz)')
# ax.set_ylabel('Probe Delay (fs)')
# fig.tight_layout()
# plt.show()

fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(omegaArrayShifted, delayArray, abs(np.fft.fftshift(dataRetrieval, axes=1)), shading = 'auto')
fig.colorbar(im)
plt.title('Measured Data')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Probe Delay (fs)')
fig.tight_layout()
plt.show()

#%%
''' Initialize data '''

# start_initialize = time.perf_counter()

## Time resolution
dt = timeArray[1] - timeArray[0]

## Dimensions of certain arrays
N = len(timeArray) # Probe length
K = int(round((max(delayArray) - min(delayArray))/dt) + N + 1) # Switch length
J = len(delayArray)

## Calculate index values of switch times closest to the delay times
closestList = np.array([ff.find_nearest(switchTimeArray, delay) for delay in delayArray])
closestIdx = closestList[:, 0].copy()
closestValue = closestList[:, 1].copy()

## Initialize probe
# For the initial probe guess, I should take a row of the data where
# the probe delay is a large negative value and fft.
guess_GDD = -500 # fs2
guess_TOD = 0# fs3


# fig, ax = plt.subplots()
# ax.plot(np.real(probeGuess*shift))
# plt.show()

## Alternative probe guess
# maxProbeValueExpected = max(abs(probeGuessInitial))
# probeGuess = maxProbeValueExpected * np.random.uniform(0, 1, size=N)
# probeFWHM = 100
# probeGuess = maxProbeValueExpected * np.exp(-4 * np.log(2) * (timeArray)**2/probeFWHM**2)

## Initialize switch. Create K matrix of ones.
## 1. Switch initial guess that is all ones.
# switchGuess = np.ones(K)

## 2. Switch initial guess that is a random complex number. The amplitude is between 0 and 1, and the phase is between 0 and 2*pi.
# switchGuess = np.random.uniform(low=0, high=1, size=K)
# switchGuess = np.random.uniform(low=0, high=1, size=K) * np.exp(-1j * np.random.uniform(low=0, high=2*np.pi, size=K))

## 3. Integrated Gaussian switch guess. This is for FROST.
## 3. Integrated Gaussian switch guess.
pumpFWHM = 20
pumpGuess = np.exp(-4 * np.log(2) * (switchTimeArray)**2/pumpFWHM**2)
switchGuessInitial = np.zeros(len(switchTimeArray))
totalIntegral = np.sum(pumpGuess)
for i in range(len(pumpGuess)):
    switchGuessInitial[i] = np.sqrt(np.sum(pumpGuess[0:len(pumpGuess)-i])/totalIntegral) # Added the sqrt because this should be the switch amplitude
switchGuess = switchGuessInitial.copy()

probeGuessInitial = np.fft.fftshift(np.fft.ifft(np.sqrt(dataRetrieval[0, :]) * np.exp(-1j*(guess_GDD/2*omegaArrayRetrieval**2 + guess_TOD/6*omegaArrayRetrieval**3)))) # Probe spectral amplitude with specified GDD and TOD
# probeGuessInitial = np.fft.fftshift(np.fft.ifft(np.sqrt(dataRetrieval[0, :]))) * np.exp(-1j * np.random.uniform(low=0, high=2*np.pi, size=N)) # Probe spectral amplitude with randomized phase
probeGuess = probeGuessInitial.copy()
## Initialize qFunction
qFunction = np.ones((J, N)) + np.zeros((J, N)) * 1j
qFunction = ff.q_update(qFunction, probeGuess, switchGuess, timeArray, delayArray, delayIdx=closestIdx)

# end_initialize = time.perf_counter()
# time_initialize = end_initialize - start_initialize

#%%
''' Run HIO algorithm '''

numHIOIterations = 15
errorList = []
diffMapErrorList = []

for i in range(numHIOIterations):
    
    # if i == 0:
    #     start_HIO_iteration = time.perf_counter()
    
    ## Calculate projections and update qFunction
    qFunctionProdProj, probeGuess, switchGuess = ff.product_projection(qFunction, probeGuess, switchGuess, timeArray, delayArray, closestIdx, timeArray, useAbsSwitch)
    qFunctionExpProj = ff.experimental_projection(2 * qFunctionProdProj - qFunction, dataRetrieval)
    qFunctionNew = (qFunction + qFunctionExpProj - qFunctionProdProj).copy()
    # qFunction = qFunction + qFunctionExpProj - qFunctionProdProj

    diffMapError = np.sqrt(np.sum(abs(qFunctionNew - qFunction)**2))
    diffMapErrorList += [diffMapError]

    qFunction = qFunctionNew.copy()

    # errorList += [ff.calculate_error(qFunction, dataRetrieval)]
    errorList += [ff.calculate_error(qFunctionProdProj, dataRetrieval)]
    
    # if i == 0:
    #     end_HIO_iteration = time.perf_counter()
    #     time_HIO_iteration = end_HIO_iteration - start_HIO_iteration

#%%
''' Plot results from HIO algorithm '''

##
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(diffMapErrorList, '.')
# ax.plot(errorList)
fig.tight_layout()
plt.show()
##

##
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.log10(errorList))
# ax.plot(errorList)
ax.set_ylabel('log$_{10}$(Err)')
# ax.set_ylabel('Err')
ax.set_xlabel('Iteration')
fig.tight_layout()
plt.show()
##

##
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionProdProj)**2)
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionProdProj))
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionExpProj))
fig.colorbar(im)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Probe Delay (fs)')
ax.set_title('Final HIO |Q$_{product}$|')
# ax.set_ylim(-50, 100)
# ax.set_xlim(-50, 50)
fig.tight_layout()
plt.show()
##

##
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionExpProj)**2)
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionProdProj))
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionExpProj))
fig.colorbar(im)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Probe Delay (fs)')
ax.set_title('Final HIO |Q$_{experiment}$|')
# ax.set_ylim(-50, 100)
# ax.set_xlim(-50, 50)
fig.tight_layout()
plt.show()
##

##
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(timeArray, delayArray, abs(qFunction)**2, shading='auto')
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionProdProj))
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionExpProj))
fig.colorbar(im)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Probe Delay (fs)')
ax.set_title('Final HIO |Q|')
# ax.set_ylim(-50, 100)
# ax.set_xlim(-50, 50)
fig.tight_layout()
plt.show()
##

#%%
''' Run ER algorithm '''
#### Implement ER algorithm

numERIterations = 3000

for i in range(numERIterations):

    # if i == 0:
    #     start_ER_iteration = time.perf_counter()

    qFunctionProdProj, probeGuess, switchGuess = ff.product_projection(qFunction, probeGuess, switchGuess, timeArray, delayArray, closestIdx, timeArray, useAbsSwitch)

    ## Some extra constraints on the switch function
    # switchGuess[switchGuess>1] = 1
    if i > 1/3 * numERIterations:
    # if i > 1/2 * numERIterations:
    # if i > 2000:
        useAbsSwitch = 0
    # useAbsSwitch = 0

    qFunctionExpProj = ff.experimental_projection(qFunctionProdProj, dataRetrieval)
    qFunction = qFunctionExpProj.copy()
    errorList += [ff.calculate_error(qFunctionProdProj, dataRetrieval)]

    if i % 100 == 0:
        print(round((i+1)/numERIterations*100, 1), '% completed.', sep='')

    # if i == 0:
    #     end_ER_iteration = time.perf_counter()
    #     time_ER_iteration = end_ER_iteration - start_ER_iteration

# endTime = time.perf_counter()
# print(f'Elapsed time is {round(endTime - startTime, 1)} s.')

#%%
''' Evaluate times '''

# print(f'Load data: {time_load_data} s')
# print(f'Initialize data: {time_initialize} s')
# print(f'HIO iteration: {time_HIO_iteration} s')
# print(f'ER iteration: {time_ER_iteration} s')


#%%
''' Plot results '''
##
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(np.log10(errorList))
# ax.plot(errorList)
# ax.set_title('Error')
ax.set_ylabel('log$_{10}$(Err)')
# ax.set_ylabel('Err')
ax.set_xlabel('Iteration')
fig.tight_layout()
plt.show()
##

#%%
print('qFunctions after ER algorithm:')

omegaArray0 = omegaArray.copy()
delayArray0 = delayArray.copy()
data0 = data.copy()

## qFunction output plot from ER algorithm
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(timeArray, delayArray, abs(qFunction)**2)
# im = ax.pcolormesh(timeArray, delayArray, abs(qFunctionProdProj)**2)
# im = ax.pcolormesh(timeArray, delayArray, (qFunctionExpProj*shift).real)
ax.set_title('|Q(t, t$_0$)|$^2$')
fig.colorbar(im)
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Probe Delay (fs)')
# ax.set_ylim(-50, 100)
# ax.set_xlim(-200, 200)
fig.tight_layout()
plt.show()
##

omegaPlot = np.fft.fftshift(omegaArrayRetrieval + omegaArrayCenter)
# qFunctionFourierInt = abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(qFunction, axes=1), axis=1), axes=1))**2
qFunctionFourierInt = abs(np.fft.fftshift(np.fft.fft(np.fft.ifftshift(qFunctionProdProj, axes=1), axis=1), axes=1))**2
dataRetrievalPlot = np.fft.fftshift(dataRetrieval, axes=1)
omegaWindow = (omegaPlot > min(omegaArray0)) & (omegaPlot < max(omegaArray0))

## qFunction output plot from ER algorithm
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(omegaPlot[omegaWindow], delayArray, abs(qFunctionFourierInt[:, omegaWindow]), shading='auto')
# im = ax.pcolormesh(omegaArray, delayArray, abs(np.fft.fft( qFunctionExpProj ))**2)
fig.colorbar(im)
ax.set_title('Retrieved Trace |Q($\omega$, t$_0$)|$^2$')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Probe Delay (fs)')
fig.tight_layout()
plt.show()
##

## Measured trace
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(omegaPlot[omegaWindow], delayArray, np.abs(dataRetrievalPlot[:, omegaWindow]), shading='auto')
fig.colorbar(im)
plt.title('Interpolated Measured Trace |Q($\omega$, t$_0$)|$^2$')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Probe Delay (fs)')
fig.tight_layout()
plt.show()
##

## Measured trace // from original data
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(omegaArray0, delayArray0, abs(data0), shading='auto')
fig.colorbar(im)
plt.title('Original Measured Trace |Q($\omega$, t$_0$)|$^2$')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Probe Delay (fs)')
fig.tight_layout()
plt.show()
##

#%%

# from matplotlib import cm
# from matplotlib.colors import ListedColormap, LinearSegmentedColormap

# jet = cm.get_cmap('jet', 2**12)
# jet_transparent_colors = jet(np.linspace(0, 1, 2**14))
# jet_white_colors = jet(np.linspace(0, 1, 2**14))
# white = np.array([1, 1, 1, 1])
# rng = 2400
# for n in range(rng):
#     weight = (n/rng)**(1/2)
#     jet_white_colors[n, :] = (weight*jet_white_colors[n, :] + (1-weight)*white)
#     jet_transparent_colors[n, 0:3] = (weight*jet_white_colors[n, 0:3] + (1-weight)*white[0:3])

# rng2=600
# for n in range(rng2):
#     weight = n/rng2
#     jet_transparent_colors[n, 3] = weight

# jet_transparent = ListedColormap(jet_transparent_colors)

#%%
''' Plot traces for comparison to Leblanc et al. '''

# import seaborn as sns

# sns.set_context('poster', font_scale=1.5)

# ## Measured trace // from original data
# fig, axs = plt.subplots(figsize=(12, 5), ncols=2, sharey=True)

# im1 = axs[0].pcolormesh(delayArray0, omegaArray0, data0.T/np.max(data.T), cmap=jet_transparent)
# im2 = axs[1].pcolormesh(delayArray, omegaPlot[omegaWindow], qFunctionFourierInt[:, omegaWindow].T/np.max(qFunctionFourierInt[:, omegaWindow].T), cmap=jet_transparent)


# cbar = fig.colorbar(im2, ax=axs[:], location='top', shrink=0.5, ticks=[0,1])

# axs[0].set_ylabel('$\omega$ (PHz)')
# axs[0].set_xlabel('t (fs)')
# axs[1].set_xlabel('t (fs)')

# # axs[0].set_ylim(0.14, 0.35)

# # plt.text(90, 0.31, "{:.2f}".format(errorList[-1]*100) + '%', fontsize=32, color='black')
# plt.text(90, 0.55, "{:.2f}".format(errorList[-1]*100) + '%', fontsize=32, color='black')


# # fig.tight_layout()
# plt.show()
# ##

#%%
'''Save trace data'''

# saveTrace = True

# if saveTrace:

#     retrievalTraceDict = {'trace_ptycho': qFunctionFourierInt, 'V_delais': delayArray, 'V_freq': omegaPlot}
#     interpTraceDict = {'trace_ptycho': dataRetrievalPlot, 'V_delais': delayArray, 'V_freq': omegaPlot}

# sio.savemat(path + folder + filename + '_DAH_retrieval.mat', {'M_trace': retrievalTraceDict})
# sio.savemat(path + folder + filename + '_DAH_interp.mat', {'M_trace': interpTraceDict})


#%%
''' Check probe and switch pulses '''

## Create time windows for plotting probe and switch pulses
probeIntNorm = abs(probeGuess)**2/max(abs(probeGuess)**2)
tauEquiv = integrate.trapz(probeIntNorm, timeArray)   # Use the equivalent pulse width measurement on pg 31 of Trebino's book
windowTimeInt = (timeArray > -4*tauEquiv) & (timeArray < 4*tauEquiv)
windowSwitchTimeInt = (switchTimeArray > -4*tauEquiv) & (switchTimeArray < 4*tauEquiv)
windowTimePhase = (probeIntNorm > 0.01)

## probe and switch output plots
probeGuessInitialIntNorm = abs(probeGuessInitial)**2/max(abs(probeGuessInitial)**2)

fig, ax = plt.subplots()
ax.plot(timeArray, probeGuessInitialIntNorm, '.', label='Initial')
ax.plot(timeArray, probeIntNorm, '.-', label='Final')
plt.show()
#### Commented out on 10/26/2021
# if useSimulatedData:
#     ax.plot(timeArray0, abs(probeFunction0)**2/max(abs(probeFunction0)**2), label='Simulation')
plt.legend()
ax.set_title('Comparison of Probe Intensity Profiles')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Probe Intensity (A.U.)')
#ax.set_xlim(-200, 200)
#ax.set_xlim(-50, 2fs200)

half_max = max(probeIntNorm) / 2
above_half = probeIntNorm > half_max
fwhm = timeArray[np.where(above_half)[0][-1]] - timeArray[np.where(above_half)[0][0]]
print('FWHM: ' + str(fwhm))


##

switchGuessIntNorm = abs(switchGuess)**2/max(abs(switchGuess)**2)

##
fig, ax = plt.subplots()
ax.plot(switchTimeArray, switchGuessIntNorm, '-')
ax.set_title('Switch Intensity Profile')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Switch Intensity (A.U.)')

#ax.set_xlim(-400, 400)
#ax.set_ylim(0, 0.005)
plt.show()

'''
fig, ax = plt.subplots()
ax.plot(switchTimeArray, abs(switchGuess)**2, '-')
ax.set_title('Switch Intensity Profile')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Switch Intensity (A.U.)')
# ax.set_xlim(-200, 200)
ax.set_xlim(500, 2000)
# ax.set_ylim(-0.1, 1.1)
'''
##

#%%
''' Check switch phase '''

## Calculate center frequency of spectrum to remove carrier phase
# centerAngFreq = np.sum(omegaArray*data[0, :])/np.sum(data[0, :])
# print(centerAngFreq)

# shiftCenter = np.exp(-1j * centerAngFreq * timeArray)
# switchPhaseShifted = np.unwrap(2 * np.angle((probeGuess * np.conjugate(shiftCenter)))) / 2

# switchPhase = -np.unwrap(2 * np.angle(switchGuess)) / 2
switchPhase = -np.unwrap(np.angle(switchGuess))

## Plot probe phase with carrier phase removed
fig, ax = plt.subplots()
ax.plot(switchTimeArray[windowSwitchTimeInt], switchPhase[windowSwitchTimeInt], '.-')
# ax.plot(timeArray, phaseShifted0, '--')
ax.set_title('Switch Phase')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Phase (rad)')
# ax.set_xlim(-400, 400)
##
plt.show()

#%%
''' Check probe phase '''

## Need to confirm phase sign in time domain and frequency domain
## I believe it is:
## TIME DOMAIN: np.exp(-1j * omega * t) except for carrier wave
## FREQ DOMAIN: np.exp(-1j * omega * t)
## I think this is the conjugate of the convention.

# phase = np.unwrap(np.angle(probeGuess)) # Old calculation
phase = -np.unwrap(np.angle(probeGuess))
phaseOffset = phase[len(phase)//2]

fig, ax = plt.subplots()
ax.plot(timeArray, phase - phaseOffset, '.-', label='Retrieval')

### Commented out on 10/26/2021
# if useSimulatedData:
#     # shift0 = np.exp(-1j * omegaArrayCenter * timeArray0) # Old clcaultion ## This shift removes an oscillatory part of the probeFunction0 to shift the frequency to compare with retrieval
#     # phase0 = np.unwrap(2 * np.angle((probeFunction0*np.conjugate(shift0)))) / 2 # Old calculation
#     omegaArrayCenter0 = (omegaArray0[0] + omegaArray0[-1])/2
#     shift0 = np.exp(1j * omegaArrayCenter0 * timeArray0)
#     # shift0 = np.exp(1j * (omegaArrayCenter0 - omegaArrayCenter) * timeArray0) ## This shift removes an oscillatory part of the probeFunction0 to shift the frequency to compare with retrieval
#     shiftDiff = np.exp(1j * (omegaArrayCenter - omegaArrayCenter0) * timeArray0)
#     phase0 = -np.unwrap(np.angle(probeFunction0 * np.conjugate(shift0*shiftDiff)))
#     phase0Offset = phase0[len(phase0)//2]
#     ax.plot(timeArray0, phase0 - phase0Offset, label='Simulation')
#     plt.legend()

# ax.plot(timeArray, np.unwrap(np.angle((probeGuess))))
# ax.plot(timeArray, np.angle((probeGuess*shift)))
ax.set_title('Probe Phase')
ax.set_xlabel('Time (fs)')
ax.set_ylabel('Phase (rad)')
#ax.set_xlim(-200, 200)
#ax.set_ylim(-20, 20)
plt.show()
##

#### I shouldn't need to do this because the carrier phase should be zero
# ## Calculate center frequency of spectrum to remove carrier phase
# # centerAngFreq = np.sum(omegaArray*data[0, :])/np.sum(data[0, :])
# centerAngFreq = np.sum(omegaArrayShifted*data[0, :])/np.sum(data[0, :])
# # find center from original data
# # centerAngFreq = np.sum(omegaArray0*data0[0, :])/np.sum(data0[0, :])
# print(centerAngFreq)

# shiftCenter = np.exp(1j * centerAngFreq * timeArray)
# # phaseShifted = np.unwrap(2 * np.angle(probeGuess * np.conjugate(shiftCenter))) / 2
# phaseShifted = -np.unwrap(np.angle(probeGuess * np.conjugate(shiftCenter))) # Negative from E(t) ~ np.exp(1j * ( omega0*t - phi(t) )) as in Trebino's book
# # phaseShifted = np.unwrap(np.angle(probeGuess))
# # phaseShifted0 = np.unwrap(np.angle((probeGuess * np.conjugate(shiftCenter))))


# ## Plot probe phase with carrier phase removed
# fig, ax = plt.subplots()
# ax.plot(timeArray[windowTimeInt], phaseShifted[windowTimeInt], '.-')
# # ax.plot(timeArray, np.unwrap(-(np.log(probeGuess * shiftCenter)).imag))
# # ax.plot(timeArray, phaseShifted0, '--')
# ax.set_title('Probe Phase')
# ax.set_xlabel('Time (fs)')
# ax.set_ylabel('Phase (rad)')
# # ax.set_xlim(-100, 100)
# # ax.set_xlim(-200, 200)
# ##

#%%
## Plot spectrum to determine if we can do phase blanking
fig, ax = plt.subplots()
ax.plot(omegaArray, data[0, :]/max(data[0, :]))
ax.set_title('Spectral Intensity')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Amplitude (A.U.)')
plt.show()

# window = (omegaArray > 2) & (omegaArray < 3) # Frequency window for 0.8 micron probe
# window = (omegaArray > 0.75) & (omegaArray < 1.75) # Frequency window for 1.8 micron probe
# window = (omegaArray > 0.3) & (omegaArray < 0.6) # Frequency window for 4 micron probe
# window = (omegaArray > 0.1) & (omegaArray < 0.5) # Frequency window for 10 micron probe

## Window based on spectral intensity
# windowFreq = (data[0, :]/max(data[0, :]) > 0.01)
windowFreq = (omegaArray>-0.18)&(omegaArray<0.25)

## Plot spectrum to determine if we can do phase blanking
fig, ax = plt.subplots()
ax.plot(omegaArray[windowFreq], data[0, windowFreq]/max(data[0, windowFreq]), '.-')
ax.set_title('Spectral Intensity')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Amplitude (A.U.)')
plt.show()

#%% The stuff toward the outside of the retrieved pulse that doesn't exist is creating those ridges.

fig, ax = plt.subplots()
ax.plot(timeArray, abs(probeGuess)**2)
plt.xlim(-100,100)
plt.title("Reconstructed Probe")
plt.show()

fig, ax = plt.subplots()
ax.plot(timeArray, np.unwrap(-np.angle(probeGuess)))
plt.show()

probePhaseAnalysis = probeGuess.copy()
#probePhaseAnalysis[(timeArray<-250)|(timeArray>250)] = 0
# probePhaseAnalysis[(timeArray<-500)|(timeArray>500)] = 0

fig, ax = plt.subplots()
ax.plot(timeArray, abs(probePhaseAnalysis)**2)
plt.show()

probeFourierPhase = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(probePhaseAnalysis)))
phaseFourier = np.unwrap(-np.angle(probeFourierPhase))

fig, ax = plt.subplots()
ax.plot(omegaArray, abs(probeFourierPhase)**2)

fig, ax = plt.subplots()
ax.plot(omegaArray, phaseFourier)
ax.set_xlim(-0.2, 0.2)
ax.set_ylim(20,40)
ax.set_title("Phase of Reconstructed Pulse")
ax.set_xlabel("$\omega$ (PHz)")
ax.set_ylabel("$\phi$ (rad)")
# ax.set_ylim(-400, -200)
plt.show()
#%%
''' Calculate phase in frequency domain and plot results '''
##
# phase, GD, GDD, TOD = ff.calculate_phase_fourier(probeGuess, omegaArray)
phase, GD, GDD, TOD = ff.calculate_phase_fourier(probePhaseAnalysis, omegaArray)
# phase, GD, GDD, TOD = ff.calculate_phase_fourier( probeGuess*np.conjugate(shift), omegaArray )
##

print('Probe phase in the frequency domain:')

##
fig, ax = plt.subplots()
ax.plot(omegaArray[windowFreq], phase[windowFreq], '.-')
# ax.plot(omegaArray, GD, '.-')
# ax.set_xlim(1.5, 3.5) # Frequency range for 0.8 micron probe
# ax.set_xlim(0.75, 1.75)
# ax.set_xlim(0.3, 0.6) # Frequency range for 4 micron probe
# ax.set_xlim(0.1, 0.5) # Frequency range for 10 micron probe
# ax.set_ylim(-5000, 5000)
ax.set_ylabel('Phase (rad)')
ax.set_xlabel('$\omega$ (PHz)')
plt.show()
##

##
fig, ax = plt.subplots()
ax.plot(omegaArray[windowFreq], GD[windowFreq], '.-')
# ax.plot(omegaArray, GD, '.-')
# ax.set_xlim(1.5, 3.5) # Frequency range for 0.8 micron probe
# ax.set_xlim(0.75, 1.75)
# ax.set_xlim(0.3, 0.6) # Frequency range for 4 micron probe
# ax.set_xlim(0.1, 0.5) # Frequency range for 10 micron probe
# ax.set_ylim(-5000, 5000)
ax.set_ylabel('GD (fs)')
ax.set_xlabel('$\omega$ (PHz)')
plt.show()
##

##
fig, ax = plt.subplots()
ax.plot(omegaArray[windowFreq], GDD[windowFreq], '.-')
# ax.plot(omegaArray, GDD, '.-')
# ax.set_xlim(1.5, 3.5) # Frequency range for 0.8 micron probe
# ax.set_xlim(0.75, 1.75) # Frequency range for 1.8 micron probe
# ax.set_xlim(0.3, 0.6) # Frequency range for 4 micron probe
# ax.set_xlim(0.1, 0.5) # Frequency range for 10 micron probe
ax.set_ylim(-5000, 5000)
ax.set_ylabel('GDD (fs$^2$)')
ax.set_xlabel('$\omega$ (PHz)')
plt.show()
##

#%%
#plot retrieved trace and GD side by side
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.pcolormesh(omegaPlot[omegaWindow], delayArray, abs(qFunctionFourierInt[:, omegaWindow]), shading='auto')
# im = ax.pcolormesh(omegaArray, delayArray, abs(np.fft.fft( qFunctionExpProj ))**2)
fig.colorbar(im)
ax.set_title('Retrieved Trace |Q($\omega$, t$_0$)|$^2$')
ax.set_xlabel('$\omega$ (PHz)')
ax.set_ylabel('Probe Delay (fs)')
fig, ax = plt.subplots()
ax.plot(omegaArray[windowFreq], GD[windowFreq], '.-')
plt.show()
# ax.plot(omegaArray, GD, '.-')
# ax.set_xlim(1.5, 3.5) # Frequency range for 0.8 micron probe
# ax.set_xlim(0.75, 1.75)
# ax.set_xlim(0.3, 0.6) # Frequency range for 4 micron probe
# ax.set_xlim(0.1, 0.5) # Frequency range for 10 micron probe
# ax.set_ylim(-5000, 5000)
ax.set_ylabel('GD (fs)')
ax.set_xlabel('$\omega$ (PHz)')
plt.show()

fig.tight_layout()
plt.show()
#%%
''' Fit polynomial to phase in frequency domain between 2 and 3 PHz '''

speedLight = 299792458
wavFit = 3000 # Wavelength around which to fit phase
angFreqFit = 2*np.pi*speedLight/(wavFit*10**-9)*10**-15
diffAngFreq = angFreqFit-carrierAngFreq
omegaFit = omegaArray+diffAngFreq
# omegaFit = omegaArray.copy()

phaseFit = np.polyfit(omegaFit[windowFreq], phase[windowFreq], 5)
fitFunction = np.poly1d(phaseFit)
# print(phaseFit)

print('GDD:', 2*phaseFit[-3], 'fs**2')
print('TOD:', 6*phaseFit[-4], 'fs**3')
print('FOD:', 24*phaseFit[-5], 'fs**4')
print('5OD:', 24*5*phaseFit[-6], 'fs**5')
# print('6OD:', 24*5*6*phaseFit[-7], 'fs**6')

print('Polynomial coefficients (highester order first):', phaseFit)

fig, ax = plt.subplots()
ax.plot(omegaFit[windowFreq], phase[windowFreq])
ax.plot(omegaFit[windowFreq], fitFunction(omegaFit[windowFreq]), '--')
plt.show()

fig, ax = plt.subplots()
ax.plot(2*np.pi*speedLight/(omegaFit[windowFreq]+carrierAngFreq), phase[windowFreq])
ax.set_xlabel("Wavelength")

ax.plot(2*np.pi*speedLight/(omegaFit[windowFreq]+carrierAngFreq), fitFunction(omegaFit[windowFreq]), '--')
plt.show()

#%%
''' Save probe and switch retrieval data '''

# probeDict = dict({'Time (fs)': timeArray, 'Probe Intensity (A.U.)': abs(probeGuess)**2/max(abs(probeGuess)**2), 'Phase (rad)': phase})
# switchDict = dict({'Time (fs)': switchTimeArray, 'Switch Intensity (A.U.)': abs(switchGuess)**2, 'Phase (rad)': switchPhase})
# errorDict = dict({'Log10 Error': np.log10(errorList)})

# dfProbeOut = pd.DataFrame(probeDict)
# dfSwitchOut = pd.DataFrame(switchDict)
# dfError = pd.DataFrame(errorDict)

identifier = 1
saveFile = True

if saveFile:
    # dfProbeOut.to_csv(path + folder + 'Probe_' + str(identifier) + '_' + filename + '.txt', index=False)
    # dfSwitchOut.to_csv(path + folder + 'Switch_' + str(identifier) + '_' + filename + '.txt', index=False)
    # dfError.to_csv(path + folder + 'Error_' + str(identifier) + '_' + filename + '.txt', index=False)
    
    
    dataDict = dict({'Probe Time (fs)': timeArray, 'Retrieved Probe': probeGuess,
                     'Switch Time (fs)': switchTimeArray, 'Retrieved Switch': switchGuess,
                     'Ang Freq (PHz)': omegaArray, 'Carrier Ang Freq (PHz)': carrierAngFreq,
                     'Delay (fs)': delayArray,
                     'Retrieved Trace': qFunctionFourierInt,
                     'Measured Trace': dataRetrievalPlot,
                     'Error': errorList,
                     'Group Delay (fs)': GD
                     })
    
    ## use pickle to save omegaArray, carrierAngFreq, timeArray, retrievedprobe, error values, switchTimeArray, Retrieval
    filename_save = path + folder + 'FROST_retrieval_data_improved_phi_guess' + str(identifier) + '.pkl'
    with open(filename_save, 'wb') as f:
        pickle.dump(dataDict, f)
        
    
