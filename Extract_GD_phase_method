# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 16:46:15 2026

@author: Connor Davis
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import scipy.signal as signal
import scipy.interpolate as interpolate
import scipy
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
from scipy.integrate import cumulative_trapezoid

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

def load_spectra_frost(fileList):
    spectrumList = []
    for file in fileList:
        spectrum = np.array(np.loadtxt(file, delimiter='\t', usecols= 1 , skiprows=1))
        spectrumList += [spectrum]
    return spectrumList


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

#%% Load trace data
''' Import files '''

# trace_path = 'C:/Users/dylan/Documents/myGitHub/FROSt/data/20220325 SHA/Scans/'
trace_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260323 FROSt Iris/Scans/'

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
wavelength_path = 'C:/Users/Connor Davis/Documents/Research/FROSt/data/20260323 FROSt Iris/wavelengths/'

#### Load multiple wavelength ranges if more than one scan is required to measure a single trace.
wavFileList = []
for file in os.listdir(wavelength_path):
    if file.endswith(".txt"):
        wavFileList += [wavelength_path + file]
wavelengthList = load_wavelengths_frost(wavFileList)

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
    
    spectrum = np.array(spectrum)
    spectrum = spectrum.flatten() #the spectrum array is concatenated 

#%% Splice spectra together for each delay/position value
#### This is for splicing more than one scan together if more than one is required for complete data set.

#### Plot data of selected trace 
traceSelect = 0
fig, ax = plt.subplots()
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Wavelength (nm)")
ax.pcolormesh(timeList[traceSelect], wavelengthList[traceSelect], traceList[traceSelect], cmap=jet_transparent)
plt.show() 


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
ax.pcolormesh(delay_data, wavelength_data, trace_data, cmap=jet_transparent)
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Wavelength (nm)")
plt.show()


cutoff_wavelengths =  [int(0.04*len(wavelength_data)), int(0.882*len(wavelength_data))]
wavelength_data = wavelength_data[cutoff_wavelengths[0]:cutoff_wavelengths[1]]
spectrum = spectrum[cutoff_wavelengths[0]:cutoff_wavelengths[1]]
trace_data = trace_data[cutoff_wavelengths[0]:cutoff_wavelengths[1],:]
fig, ax = plt.subplots()
ax.pcolormesh(delay_data, wavelength_data, trace_data, cmap=jet_transparent)
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Wavelength (nm)")
ax.set_title("FROSt Depletion")
plt.show()


#%%


lam_nm = np.asarray(wavelength_data, float)   # (Nlam,)
t_fs   = np.asarray(delay_data, float)        # (Nt,)
S      = np.asarray(trace_data, float)        # (Nlam, Nt)


def sigmoid(t, a, b, t0, w):
    # a + b/(1 + exp((t-t0)/w))
    return a + b / (1.0 + np.exp((t - t0)/w))

smooth_window = 25  # odd number
polyorder = 3
fit_half_window = 100 # fs

tau0_fs = np.full(len(lam_nm), np.nan)
fit_ok  = np.zeros(len(lam_nm), dtype=bool)



for i in range(len(lam_nm)):
    y = S[i, :].astype(float)

    # smooth
    if smooth_window < len(t_fs):
        y_s = savgol_filter(y, smooth_window, polyorder, mode='interp')
    else:
        y_s = y

    # max dy/dt for initial guess
    dy = np.gradient(y_s, t_fs)
    j0 = np.argmin(dy)
    t0_guess = t_fs[j0]

    # fit window
    m = (t_fs > t0_guess - fit_half_window) & (t_fs < t0_guess + fit_half_window)
    if np.sum(m) < 12:
        tau0_fs[i] = t0_guess
        continue

    t_fit = t_fs[m]
    y_fit = y_s[m]

    # initial guess（a:right height，b:left height-right height，w:edge width）
    a0 = np.median(y_fit[t_fit > t0_guess]) if np.any(t_fit > t0_guess) else np.median(y_fit)
    b0 = np.median(y_fit[t_fit < t0_guess]) - a0 if np.any(t_fit < t0_guess) else (np.max(y_fit)-np.min(y_fit))
    w0 = 20.0

    try:
        popt, _ = curve_fit(sigmoid, t_fit, y_fit, p0=[a0, b0, t0_guess, w0], maxfev=3000)
        tau0_fs[i] = popt[2]
        fit_ok[i] = True
    except Exception:
        tau0_fs[i] = t0_guess

plt.figure()
plt.plot(lam_nm, tau0_fs, '.', ms=2)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Edge time tau0 (fs)")
plt.title("Extracted tau0(lambda) from FROSt (sigmoid fit)")
plt.show()



lam = lam_nm.copy()          # nm
tau = tau0_fs.copy()         # fs



# relative group delay
#tau = tau - np.mean(tau)   # fs

c = 299792458.0
omega = 2*np.pi*c/(lam*1e-9)   # rad/s

p = np.argsort(omega)
omega = omega[p]
tau_s = (tau[p]) * 1e-15       # fs -> s


# smooth window
w = 25
if w >= len(tau_s):
    w = len(tau_s)//2*2 - 1  
if w < 9:
    w = 9 

tau_s_sm = savgol_filter(tau_s, w, 3, mode='interp')


phi = cumulative_trapezoid(tau_s_sm, omega, initial=0.0)  # rad

# delete constant and linear item
#q = np.polyfit(omega, phi, 1)
#phi_flat = phi - (q[0]*omega + q[1])


plt.figure()
plt.plot(omega, tau_s*1e15, '.', ms=2, alpha=0.3, label='raw')
plt.plot(omega, tau_s_sm*1e15, '-', lw=1.5, label='smoothed')
plt.xlabel('ω (rad/s)')
plt.ylabel('Relative group delay (fs)')
plt.legend()
plt.title('τ(ω) from FROSt')
plt.show()
phase = phi
plt.figure()
plt.plot(omega, phi, '-')
plt.xlabel('ω (rad/s)')
plt.ylabel('Spectral phase φ(ω) (rad)')
plt.title('Recovered phase from τ(ω)')
plt.show()

fig, ax = plt.subplots()
ax.set_xlabel("Delay (fs)")
ax.set_ylabel("Wavelength (nm)")
ax.set_title("FROSt Depletion")
ax.pcolormesh(delay_data, wavelength_data, trace_data, cmap=jet_transparent)
plt.plot(tau_s_sm*1e15, 2*np.pi*c/omega*1E9, '-', lw=1.5, label='smoothed')
plt.show()


#%%
plt.plot(2*np.pi*c/omega*1e9, phi)
plt.xlabel('Wavelength (nm)')
plt.ylabel('Spectral phase φ($\lambda$) (rad)')
plt.show()


#%%
pump_wl = 1030E-9
seed_wls = 1/(omega/(2*np.pi*c)+1/pump_wl)
plt.plot(seed_wls*1e9, phi)
plt.xlabel('Seed Wavelength (nm)')
plt.ylabel('Spectral phase φ($\lambda$) (rad)')
plt.show()
#%%
'''
np.savetxt(
    "C:/Users/Connor Davis/Documents/Research/DAZZLER/Mid-IR FROSt Compensation Code/Extracted Phases/phi_raw_vs_omega.txt",
    np.column_stack([seed_wls[::-10]*1e9, phi[::-10]]),
    fmt="%.6f"
)
'''
#%%
DazzlerWavelength = 743.5E-9
centerFreq = 2*np.pi*c/DazzlerWavelength-2*np.pi*c/1030E-9

phaseFit = np.polyfit((omega-centerFreq)*1E-15, phi, 4)
fitFunction = np.poly1d(phaseFit)
# print(phaseFit)

plt.plot((omega-centerFreq)*1E-15, phi, label = "Extracted Phase")
plt.plot((omega-centerFreq)*1E-15, fitFunction((omega-centerFreq)*1E-15), label = "Polynomial Fit")
plt.xlabel("$\omega$ (PHz), relative to DAZZLER central wavelength")
plt.ylabel("$\phi$ (rad)")
plt.legend()
plt.show()

print('GDD:', 2*phaseFit[-3], 'fs**2')
print('TOD:', 6*phaseFit[-4], 'fs**3')
print('FOD:', 24*phaseFit[-5], 'fs**4')
print('5OD:', 24*5*phaseFit[-6], 'fs**5')

#%%
# -----------------------------
# INPUTS
# -----------------------------
# omega     : angular frequency array (uniform)
# phi       : measured phase
# spectrum  : measured spectral intensity

# -----------------------------
# PREPROCESS
# -----------------------------
spectrum = spectrum - np.min(spectrum)
spectrum[spectrum < 0] = 0

A = np.sqrt(spectrum)

# -----------------------------
# COMPLEX SPECTRA
# -----------------------------
E_omega    = A * np.exp(1j * phi)
E_omega_TL = A

# -----------------------------
# ZERO PADDING
# -----------------------------
pad_factor = 8   # increase for higher time sampling

N = len(E_omega)
N_pad = pad_factor * N

# Allocate padded arrays
Ew_pad    = np.zeros(N_pad, dtype=complex)
Ew_TL_pad = np.zeros(N_pad, dtype=complex)

# Center the spectrum before inserting
Ew_shift    = np.fft.fftshift(E_omega)
Ew_TL_shift = np.fft.fftshift(E_omega_TL)

start = (N_pad - N) // 2
Ew_pad[start:start+N]    = Ew_shift
Ew_TL_pad[start:start+N] = Ew_TL_shift

# Shift back before IFFT
Ew_pad    = np.fft.ifftshift(Ew_pad)
Ew_TL_pad = np.fft.ifftshift(Ew_TL_pad)

# -----------------------------
# IFFT
# -----------------------------
E_t    = np.fft.fftshift(np.fft.ifft(Ew_pad))
E_t_TL = np.fft.fftshift(np.fft.ifft(Ew_TL_pad))

# -----------------------------
# TIME AXIS
# -----------------------------
domega = omega[1] - omega[0]

T = 2 * np.pi / domega   # same total time window
dt = T / N_pad

t = np.linspace(-T/2, T/2, N_pad)

# -----------------------------
# NORMALIZATION
# -----------------------------
E_t    /= np.max(np.abs(E_t))
E_t_TL /= np.max(np.abs(E_t_TL))

I_t    = np.abs(E_t)**2
I_t_TL = np.abs(E_t_TL)**2

# -----------------------------
# PLOT
# -----------------------------
plt.figure(figsize=(10,5))

plt.plot(t*1E15, I_t, label='Measured Pulse')
plt.plot(t*1E15, I_t_TL, '--', label='Transform-Limited')

plt.xlabel('Time (fs)')
plt.ylabel('Normalized Intensity')
plt.title('Pulse in the Time Domain')
plt.legend()
plt.grid()
plt.xlim(-200,200)
plt.show()

def fwhm(t, I):
    I = I / np.max(I)
    inds = np.where(I > 0.5)[0]
    return t[inds[-1]] - t[inds[0]]

print("Measured FWHM (fs):", fwhm(t, I_t)*1E15)
print("TL FWHM (fs):", fwhm(t, I_t_TL)*1E15)

