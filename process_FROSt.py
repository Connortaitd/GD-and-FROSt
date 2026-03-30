# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:56:50 2020

@author: dylan
"""
import sys

#### Add 'FROSt/code/Process traces/' and 'FROSt/data/' to python path. ####
# sys.path.insert(0, 'C:/Users/dylan/Documents/myGitHub/FROSt/code/Process traces/')
# sys.path.insert(1, 'C:/Users/dylan/Documents/GitHub/FROSt/code/Process traces/')
# sys.path.insert(2, 'C:/Users/dylan/Documents/myGitHub/FROSt/data/')
# sys.path.insert(3, 'C:/Users/dylan/Documents/GitHub/FROSt/data/')

#sys.path.insert(0, 'C:/Users/D/Documents/myGitHub/FROSt/code/Process traces/')
#sys.path.insert(1, 'C:/Users/D/Documents/myGitHub/FROSt/data/')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy

#%%
#### Create custom colormap
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

jet = cm.get_cmap('jet', 2**12)
jet_transparent_colors = jet(np.linspace(0, 1, 2**14))
jet_white_colors = jet(np.linspace(0, 1, 2**14))
white = np.array([1, 1, 1, 1])
rng = 2400  
for n in range(rng):
    weight = (n/rng)**(1/2)
    jet_white_colors[n, :] = (weight*jet_white_colors[n, :] + (1-weight)*white)    
    jet_transparent_colors[n, 0:3] = (weight*jet_white_colors[n, 0:3] + (1-weight)*white[0:3])

rng2=600
for n in range(rng2):
    weight = n/rng2
    jet_transparent_colors[n, 3] = weight
   
jet_transparent = ListedColormap(jet_transparent_colors)

#%% Define some functions
def load_data_frost(fileList):

    positionList = []
    columnList = []
    traceList = []
    timeList = []

    for file in fileList:
        data = np.loadtxt(file, delimiter='\t')

        # First column is time in fs
        time_fs = data[:, 0][0:-1]    
        
        # Remaining columns are traces
        trace = np.nan_to_num(data[:-1, 1:],nan=0)

        # Column indices (same role as before)
        column = np.arange(trace.shape[1])

        # No position info in new format → keep placeholder
        position = time_fs*299792458*1e-15*1e3

        positionList.append(position)
        timeList.append(time_fs)
        columnList.append(column)
        traceList.append(trace)
        
    for i in range(0, len(traceList)):
        traceList[i] = (traceList[i]).T
        
    return positionList, timeList, columnList, traceList

def load_wavelengths_frost(fileList):
    wavelengthList = []
    for file in fileList:
        wavelength = np.array(np.loadtxt(file, delimiter='\t', usecols=[0], skiprows=1))        
        wavelengthList += [wavelength]
    return wavelengthList

def wav_to_angfreq(wavelength, trace):
    speedLight = 299792458
    angfreq = 2*np.pi*speedLight / (wavelength*1e-9) * 1e-15 # in PHz

    #### When converting from linear wavelength grid to linear angular frequency grid, must multiply by lambda**2/(2*np.pi*c)
    nonlinearScaling = (wavelength*1e-9)**2 / (2*np.pi*speedLight)
    if len(trace.shape)>1:
        nonlinearScalingMesh = np.tile(nonlinearScaling, (trace.shape[1], 1)).transpose()
    else:
        nonlinearScalingMesh = nonlinearScaling
    traceAngFreq = nonlinearScalingMesh*trace
    
    return(angfreq, traceAngFreq)

def load_spectra_frost(fileList):
    spectrumList = []
    for file in fileList:
        spectrum = np.array(np.loadtxt(file, delimiter='\t', usecols= 1 , skiprows=1))
        spectrumList += [spectrum]
    return spectrumList

#%% Load trace data
''' Import files '''

# trace_path = 'C:/Users/dylan/Documents/myGitHub/FROSt/data/20220325 SHA/Scans/'
trace_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260227 FROSt/Scans/'

#### Load multiple files if more than one scan is required to measure a single trace.
fileList = []
for file in os.listdir(trace_path):
    if file.endswith(".scan"):
        fileList += [trace_path + file]


positionList, timeList, columnList, traceList = load_data_frost(fileList)


#### Load single wavelength array if only one scan is required to capture entire trace.
#positionList, timeList, columnList, traceList = load_data_frost([trace_path+'FROSt 3380nm Full Signal Beam.txt'])

#%% Load wavelength data
''' Import wavelength files '''
    
# wavelength_path = 'C:/Users/dylan/Documents/myGitHub/FROSt/data/20220325 SHA/'
wavelength_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260227 FROSt/wavelengths/'

#### Load multiple wavelength ranges if more than one scan is required to measure a single trace.
wavFileList = []
for file in os.listdir(wavelength_path):
    if file.endswith(".txt"):
        wavFileList += [wavelength_path + file]
wavelengthList = load_wavelengths_frost(wavFileList)

#### Load single wavelength array if only one scan is required to capture entire trace.
#wavelengthList = load_wavelengths_frost([wavelength_path+'3380 Wavelength.txt'])

#%% Load scaling data
''' Import scaling data '''

#### MCT array scaling factor updated on 3/25/2022. Data older than this uses old scaling factor files. ####
# scaling_path = 'C:/Users/dylan/Documents/myGitHub/FROSt/code/Process traces/'
scaling_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/code/Process traces/'

scalingDataFrame = pd.read_csv(scaling_path+'scaling_factor.txt', delimiter=',')

#%%
spectrum_files_present = True

if(spectrum_files_present == True):
    spectrum_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260227 FROSt/Spectra/'

    #### Load spectra if more than one scan is required to measure a single trace.
    spectrumFileList = []
    for file in os.listdir(spectrum_path):
        if file.endswith(".txt"):
            spectrumFileList += [spectrum_path + file]
    spectrum = load_spectra_frost(spectrumFileList)
    
    for i in range(0, len(traceList)):
        traceList[i] = (traceList[i]+1)*spectrum[i][:,np.newaxis]
    
    
    
#%% Splice spectra together for each delay/position value
#### This is for splicing more than one scan together if more than one is required for complete data set.

#### Plot data
traceSelect = 0
fig, ax = plt.subplots()
ax.pcolormesh(timeList[traceSelect], wavelengthList[traceSelect], traceList[traceSelect], cmap=jet_transparent)
plt.show() 



fig, ax = plt.subplots()
for ii in range(len(wavelengthList)):
    ax.plot(wavelengthList[ii], traceList[ii][:,0], 'b')
    ax.plot(wavelengthList[ii], traceList[ii][:,-1], 'k')
    # ax.plot(wavelengthList[ii], traceList[ii][:,len(timeList[0])//2-20], 'r')
    ax.plot(wavelengthList[ii], traceList[ii][:,len(timeList[0])//2], 'r')

#### Calculate fraction depletion between initial delay value and final delay value
diffList = []
for ii in range(len(wavelengthList)):
    diff = (traceList[ii][:,0] - traceList[ii][:,-1])/traceList[ii][:,0]
    diffList += [diff]
    
#### Plot depletion as a function of wavelength.
fig, ax = plt.subplots()
for ii in range(len(wavelengthList)):
    ax.plot(wavelengthList[ii], diffList[ii])
ax.set_ylim(0, 1.0)
ax.axhline(0.9, c='k', ls='--')
plt.show()
#%%
'''
Dylan's Code:
    Combine the wavelength and spectrum files into one master array, with the data overlapping
'''
'''
#### Begin splicing. Note, splicing is not required if multiple scans are not needed to capture full bandwidth.
position = positionList[0].copy()
delay = timeList[0].copy()
wavInput_List = []
yInput_List = []

for ii in range(len(position)):
    
    scaleList = np.ones(len(wavelengthList) - 1)
    # scaleList = np.zeros(wavData.shape[0] - 1)
    # scaleList[0] = 0

    ## Combine all measurements without using multipicative factor (i.e. noscale) and scale with power scaling
    ## Combine without smoothing data
    wavInput_noscale = wavelengthList[0]
    yInput_noscale = traceList[0][:,ii]
    
    for jj in range(len(wavelengthList) - 1):
        wavInput_noscale, yInput_noscale = mm.combine_arrays_noscale(wavInput_noscale, yInput_noscale, wavelengthList[jj+1], traceList[jj+1][:,ii], project=scaleList[jj])
    
    if ii == 0:
        wavelength_data = wavInput_noscale.copy()
        delay_data = delay.copy()
        position_data = position.copy()
        trace_data = np.zeros(shape=(len(wavelength_data), len(position_data)))
        
    trace_data[:,ii] = yInput_noscale.copy()
    
    # wavInput_List += [wavInput_noscale]
    # yInput_List += [yInput_noscale]
'''
#%%
'''
Connor's Code:
    Combine the wavelength and spectrum files into one master array, with the data NOT overlapping
'''
position = positionList[0].copy()
delay = timeList[0].copy()
wavInput_List = []
yInput_List = []

for ii in range(len(position)):
    
    scaleList = np.ones(len(wavelengthList) - 1)
    # scaleList = np.zeros(wavData.shape[0] - 1)
    # scaleList[0] = 0

    ## Combine all measurements without using multipicative factor (i.e. noscale) and scale with power scaling
    ## Combine without smoothing data
    wavInput_noscale = wavelengthList[0]
    yInput_noscale = traceList[0][:,ii]
    
    for jj in range(len(wavelengthList) - 1):
        wavInput_noscale = np.append(wavInput_noscale, wavelengthList[jj+1])
        yInput_noscale = np.append(yInput_noscale, traceList[jj+1][:,ii])
    if ii == 0:
        wavelength_data = wavInput_noscale.copy()
        delay_data = delay.copy()
        position_data = position.copy()
        trace_data = np.zeros(shape=(len(wavelength_data), len(position_data)))
        
    trace_data[:,ii] = yInput_noscale.copy()
    
    # wavInput_List += [wavInput_noscale]
    # yInput_List += [yInput_noscale]
#%%
fig, ax = plt.subplots()
ax.pcolormesh(position_data, wavelength_data, trace_data, cmap=jet_transparent)
plt.show()

#%% Scale data with scaleList_noscale
finterp = interpolate.interp1d(scalingDataFrame['Wavelength (nm)'], scalingDataFrame['Power Scaling'], kind='cubic')

scalingInterpolated = finterp(wavelength_data)
scaling_array = np.tile(scalingInterpolated, (len(position_data),1)).transpose()

#%% Original way to applying MCT response scaling to data

#### Scale data for MCT response above threshold and within a wavelength range
trace_data_scaled = trace_data.copy()
trace_max = np.amax(trace_data)
threshold = 1
trace_data_scaled[trace_data>threshold*trace_max] = trace_data[trace_data>threshold*trace_max] * scaling_array[trace_data>threshold*trace_max]

#### Set some spectral values to 0 if scaling factor is mostly from noise. Usually only applied for wavelengths less than 2000 nm when using the MCT array spectrometer.
# trace_data_scaled[wavelength_data<2000] = 0

fig, ax = plt.subplots()
ax.pcolormesh(position_data, wavelength_data, trace_data_scaled, cmap=jet_transparent, shading='auto')

# fig, ax = plt.subplots()
# ax.plot(wavelength_data, trace_data_scaled[:,-1])

plt.show()
#%% Filter data by filtering Fourier amplitude along both dimensions. Different filter shapes can be used.

fourier_data = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(trace_data_scaled)))

filter_type = 'Rectangular'
#### Options are 'Rectangular', 'Ellipsoid', 'Gaussian', 'None', and 'Combined Gaussian

if filter_type=='None':
    traceFiltered = trace_data_scaled.copy()
    filterNotes = ['MCT Response filter: Threshold='+str(threshold),
                  'Filter Type: '+filter_type]

elif filter_type=='Rectangular':
    
    omegaIdxWidth = 300
    delayIdxWidth = 50
    
    center0 = (trace_data_scaled.shape[0])/2 # omega center
    center1 = (trace_data_scaled.shape[1])/2 # delay center
    
    x = np.arange(len(position_data)) # indices of delay dimension
    y = np.arange(len(wavelength_data)) # indices of wavelength dimension
    
    # omega_boolean = (y<center0-omegaIdxWidth/2)|(y>center0+omegaIdxWidth/2)
    # delay_boolean = (x<center1-delayIdxWidth/2)|(x>center1+delayIdxWidth/2)
    # fourier_data_rect = fourier_data.copy()
    # fourier_data_rect[omega_boolean, delay_boolean] = 0
    
    x_grid, y_grid = np.meshgrid(x, y) # index grids
    omega_boolean = (y_grid<center0-omegaIdxWidth/2)|(y_grid>center0+omegaIdxWidth/2)
    delay_boolean = (x_grid<center1-delayIdxWidth/2)|(x_grid>center1+delayIdxWidth/2)
    fourier_data_rect = fourier_data.copy()
    fourier_data_rect[omega_boolean|delay_boolean] = 0
    
    traceRect = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_data_rect)))
    traceFiltered = abs(traceRect)

    filterNotes = ['MCT Response filter: Threshold='+str(threshold),
                  'Filter Type: '+filter_type,
                  'Filter Parameters: omegaIdxWidth='+str(omegaIdxWidth)+', delayIdxWidth='+str(delayIdxWidth)]
   
    #### Plot to view data
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)
    p0 = ax[0].pcolormesh(abs(fourier_data)/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[0].axvline(center1-delayIdxWidth/2, c='k')
    ax[0].axvline(center1+delayIdxWidth/2, c='k')
    ax[0].axhline(center0-omegaIdxWidth/2, c='k')
    ax[0].axhline(center0+omegaIdxWidth/2, c='k')
    ax[0].set_title('Fourier Amplitudes')
    p1 = ax[1].pcolormesh(~(omega_boolean|delay_boolean), cmap=jet_transparent)
    ax[1].set_title('Hyper Gaussian Filter')
    p2 = ax[2].pcolormesh(abs(fourier_data_rect)/abs(fourier_data_rect).max(), cmap=jet_transparent, vmax=0.1)
    ax[2].set_title('Filtered Fourier Amplitudes')
    plt.show()
    
elif filter_type=='Ellipsoid':
    #### Ellipsoid filter
    idxRadius0 = 25 # Wavelength
    idxRadius1 = 25 # Position/Delay
    
    ## Create boolean array
    ellipsoidRadiusSq = idxRadius0**2 + idxRadius1**2
    center0 = (trace_data_scaled.shape[0]-1)/2
    center1 = (trace_data_scaled.shape[1]-1)/2
    ellipCheck = np.ones(shape=trace_data_scaled.shape)
    
    for i in range(trace_data_scaled.shape[0]):
        for j in range(trace_data_scaled.shape[1]):
            if (i-center0)**2/idxRadius0**2 + (j-center1)**2/idxRadius1**2 > 1:
                ellipCheck[i,j] = 0
    
    traceEllipsoid = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_data*ellipCheck)))
    traceFiltered = abs(traceEllipsoid)
    
    filterNotes = ['MCT Response filter: Threshold='+str(threshold),
                  'Filter Type: '+filter_type,
                  'Filter Parameters: omegaIdxRadius='+str(idxRadius0)+', delayIdxFWHM='+str(idxRadius1)]
    
    # fig, ax = plt.subplots()
    # ax.pcolormesh(ellipCheck)
    # ax.set_xlim(75, 125)
    # ax.set_ylim(300, 450)
    
    # fig, ax = plt.subplots()
    # ax.pcolormesh(abs(fourier_data), cmap=jet_transparent)
    # ax.set_xlim(75, 125)
    # ax.set_ylim(300, 450)
    
    # fig, ax = plt.subplots()
    # ax.pcolormesh(abs(fourier_data)*ellipCheck, cmap=jet_transparent)
    # ax.set_xlim(75, 125)
    # ax.set_ylim(300, 450)
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)
    p0 = ax[0].pcolormesh(abs(fourier_data)/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[0].set_title('Fourier Amplitudes')
    p1 = ax[1].pcolormesh(ellipCheck, cmap=jet_transparent)
    ax[1].set_title('Ellipsoid Filter')
    p2 = ax[2].pcolormesh(abs(fourier_data)*ellipCheck/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[2].set_title('Filtered Fourier Amplitudes')
    plt.show()

elif filter_type=='Gaussian':
    #### Gaussian or Hyper-gaussian filter
    x = np.arange(len(position_data)) # indices of delay dimension
    y = np.arange(len(wavelength_data)) # indices of wavelength dimension
    x_grid, y_grid = np.meshgrid(x, y) # index grids
    omegaIdxFWHM = 10  # FWHM of gaussian filter along angular frequency dimension, units are integer indices
    delayIdxFWHM = 100 # FWHM of gaussian filter along delay dimension, units are integer indices # Used 75 before
    m, n = 3, 3        # order of hyper-gaussian filters (1 is normal gaussian, m is for angular frequency dim, n is for delay dim)
    gaussFilter2D = np.exp(-np.log(2)*(4*(y_grid-len(wavelength_data)/2)**2/(omegaIdxFWHM)**2)**n) * np.exp(-np.log(2)*(4*(x_grid-len(position_data)/2)**2/(delayIdxFWHM)**2)**m)
    traceGauss = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_data*gaussFilter2D)))
    traceFiltered = abs(traceGauss)
    
    filterNotes = ['MCT Response filter: Threshold='+str(threshold),
                  'Filter Type: '+filter_type,
                  'Filter Parameters: omegaIdxFWHM='+str(omegaIdxFWHM)+', delayIdxFWHM='+str(delayIdxFWHM)+', m='+str(m)+', n='+str(n)]
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)

    p0 = ax[0].pcolormesh(abs(fourier_data)/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[0].contour(gaussFilter2D, levels=[0.5], colors='k')
    ax[0].set_title('Fourier Amplitudes')
    p1 = ax[1].pcolormesh(gaussFilter2D, cmap=jet_transparent)
    ax[1].set_title('Hyper Gaussian Filter')
    p2 = ax[2].pcolormesh(abs(fourier_data)*gaussFilter2D/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[2].set_title('Filtered Fourier Amplitudes')
    plt.show()
    
elif filter_type=='Combined Gaussian':
    #### Gaussian or Hyper-gaussian filter
    x = np.arange(len(position_data)) # indices of delay dimension
    y = np.arange(len(wavelength_data)) # indices of wavelength dimension
    x_grid, y_grid = np.meshgrid(x, y) # index grids
    
    #### Filter 1
    omegaIdxFWHM_1 = 10  # FWHM of gaussian filter along angular frequency dimension, units are integer indices
    delayIdxFWHM_1 = 500 # FWHM of gaussian filter along delay dimension, units are integer indices # Used 75 before
    m_1, n_1 = 3, 3        # order of hyper-gaussian filters (1 is normal gaussian, m is for angular frequency dim, n is for delay dim)
    gaussFilter2D_1 = np.exp(-np.log(2)*(4*(y_grid-len(wavelength_data)/2)**2/(omegaIdxFWHM_1)**2)**n_1) * np.exp(-np.log(2)*(4*(x_grid-len(position_data)/2)**2/(delayIdxFWHM_1)**2)**m_1)
    
    #### Filter 2
    omegaIdxFWHM_2 = 50  # FWHM of gaussian filter along angular frequency dimension, units are integer indices
    delayIdxFWHM_2 = 50 # FWHM of gaussian filter along delay dimension, units are integer indices # Used 75 before
    m_2, n_2 = 3, 3        # order of hyper-gaussian filters (1 is normal gaussian, m is for angular frequency dim, n is for delay dim)
    gaussFilter2D_2 = np.exp(-np.log(2)*(4*(y_grid-len(wavelength_data)/2)**2/(omegaIdxFWHM_2)**2)**n_2) * np.exp(-np.log(2)*(4*(x_grid-len(position_data)/2)**2/(delayIdxFWHM_2)**2)**m_2)
    
    
    gaussFilter2D = np.maximum(gaussFilter2D_1, gaussFilter2D_2)
    # traceComGauss = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_data*gaussFilter2D_1*gaussFilter2D_2)))
    traceComGauss = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(fourier_data*gaussFilter2D)))
    traceFiltered = abs(traceComGauss)
    
    filterNotes = ['MCT Response filter: Threshold='+str(threshold),
                  'Filter Type: '+filter_type,
                  'Filter Parameters 1: omegaIdxFWHM='+str(omegaIdxFWHM_1)+', delayIdxFWHM='+str(delayIdxFWHM_1)+', m='+str(m_1)+', n='+str(n_1),
                  'Filter Parameters 2: omegaIdxFWHM='+str(omegaIdxFWHM_2)+', delayIdxFWHM='+str(delayIdxFWHM_2)+', m='+str(m_2)+', n='+str(n_2)]
    
    fig, ax = plt.subplots(ncols=3, figsize=(15, 5), sharey=True)

    p0 = ax[0].pcolormesh(abs(fourier_data)/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[0].contour(gaussFilter2D_1, levels=[0.5], colors='r')
    ax[0].contour(gaussFilter2D_2, levels=[0.5], colors='b')
    ax[0].contour(np.maximum(gaussFilter2D_1, gaussFilter2D_2), levels=[0.5], colors='k')
    ax[0].set_title('Fourier Amplitudes')
    p1 = ax[1].pcolormesh(gaussFilter2D, cmap=jet_transparent)
    ax[1].set_title('Hyper Gaussian Filter')
    p2 = ax[2].pcolormesh(abs(fourier_data)*gaussFilter2D/abs(fourier_data).max(), cmap=jet_transparent, vmax=0.1)
    ax[2].set_title('Filtered Fourier Amplitudes')
    plt.show()
else:
    print('Specified filter not recognized.')


#%% Plot raw trace and filtered traces

fig, ax = plt.subplots()
ax.pcolormesh(delay_data, wavelength_data, trace_data_scaled, cmap=jet_transparent, shading='auto')
plt.show()

fig, ax = plt.subplots()
ax.pcolormesh(delay_data, wavelength_data, traceFiltered, cmap=jet_transparent, shading='auto')
plt.show()

##### Ensure that absorption isn't affected
fig, ax = plt.subplots()
ax.plot(delay_data, trace_data_scaled[len(wavelength_data)//2, :])
ax.plot(delay_data, traceFiltered[len(wavelength_data)//2, :])
plt.show()

#%% Remove edges if affected by filter.
#### Need to implement if desired.


#%% Transform wavelength axis to angular frequency

angfreq_data, trace_angfreq = wav_to_angfreq(wavelength_data, traceFiltered)

fig, ax = plt.subplots()
ax.pcolormesh(delay_data, angfreq_data, trace_angfreq, cmap=jet_transparent, shading='auto')
plt.show()

#### Calculate carrier angular frequency
delay_value_init_spectrum = -150
spectrum_angfreq = np.average(trace_angfreq[:,delay_data<delay_value_init_spectrum], axis=1)

fig, ax = plt.subplots()
ax.plot(angfreq_data, spectrum_angfreq)
plt.show()

#### Calculation of carrier angular frequency
carrier_angfreq = np.sum(angfreq_data*spectrum_angfreq)/np.sum(spectrum_angfreq)
print('The carrier angular frqeuency of the fundamental spectrum is', carrier_angfreq, 'PHz.')

#%% Interpolate to be linear in angular frequency
print(len(angfreq_data))

# numFFT = 2**10
numFFT = 2**10 #### Keep the frequency resolution what it is from the spectrometer.
#### This can be increased to provide higher frequency resolution or larger time range.

## Calculate dt from specified value and interpolate for data
# dt = 5.0 # Time resolution in fs. Used to calculate time array and angular frequency array.
# dt = 3.0 # Try to keep time resolution as close to data as possible.
# delayArrayFFT = np.arange(-numFFT//2, numFFT//2, dtype=float) * dt

## Calculate dt from measurement or specify a specific dt depending on required time resolution.
dt = (delay_data[-1] - delay_data[0])/(len(delay_data)-1) # time resolution for probe equal to delay resolution
dt = 6
timeArrayFFT = np.arange(-numFFT//2, numFFT//2, dtype=float) * dt

# domega = 2*np.pi/(delayArrayFFT[-1] - delayArrayFFT[0])
domega = 2*np.pi/(timeArrayFFT[-1] - timeArrayFFT[0])
omegaArrayFFT = np.arange(-numFFT//2, numFFT//2, dtype=float) * domega

print('The number of points in the FFT grid is ', numFFT, '.', sep='')
print('The time resolution is ', dt, ' fs.', sep='')
print('The time range is ', timeArrayFFT[-1]-timeArrayFFT[0], ' fs.', sep='')
print('The angular frequency resolution is ', domega, ' PHz.', sep='')
print('The angular frequency range is ', omegaArrayFFT[-1]-omegaArrayFFT[0], ' PHz.', sep='')
print('The delay resolution is ', dt, ' fs.', sep='')
print('The delay range is ', delay_data[-1]-delay_data[0], ' fs.', sep='')


traceInterpFunction = scipy.interpolate.interp2d(delay_data, angfreq_data-carrier_angfreq, trace_angfreq, fill_value=0, bounds_error=False)
traceInterp = traceInterpFunction(delay_data, omegaArrayFFT)
# traceInterp = traceInterpFunction(delayArrayFFT, omegaArrayFFT)

#%%
paramList = ['Time resolution (fs): '+str(dt), 'Time range (fs): '+str(timeArrayFFT[-1]-timeArrayFFT[0]),
             'Ang freq resolution (PHz): '+str(domega), 'Ang freq range (PHz): '+str(omegaArrayFFT[-1]-omegaArrayFFT[0]),
             'Delay resolution (fs): '+str(dt), 'Delay range (fs): '+str(delay_data[-1]-delay_data[0]),
            'Array sizes: '+str(numFFT)]
print(paramList)

#%% Plot final trace

fig, ax = plt.subplots()
ax.pcolormesh(delay_data, omegaArrayFFT, traceInterp, cmap=jet_transparent, shading='auto')
ax.set_xlabel('Probe Delay (fs)')
ax.set_ylabel('Angular Frequency (PHz)')
ax.set_title('Processed Trace')
# ax.set_ylim(-0.5, 0.5)
plt.show()

#%% Save Data

import scipy.io as sio

savePath = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260227 FROSt/'
saveFolder = ''
saveFilename = 'processed trace'
dataTraceDict = {'trace': traceInterp, 'time': timeArrayFFT, 'angfreq': omegaArrayFFT, 'delay': delay_data,
                 'parameters': paramList, 'filtering': filterNotes,
                 'carrierAngFreq': carrier_angfreq}
# dataTraceDict = {'trace': traceFiltered, 'delay': delayArrayFFT, 'angfreq': omegaArrayFFT,
#                  'probe': probeTimeAmp, 'spectrumFund': spectrumAngFreqFund.to_numpy(),
#                  'parameters': paramList, 'filtering': filterNotes, 
#                  'angFreqFund': angFreqFund, 'carrierAngFreq': carrierFreqFund}
sio.savemat(savePath + saveFolder + saveFilename + '.mat', {'M_trace': dataTraceDict})