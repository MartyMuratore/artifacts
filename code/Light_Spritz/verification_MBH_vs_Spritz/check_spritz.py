# Open the HDF5 file in read mode
### search pipeline for 

from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove

import numpy as np
import os
import sys
import h5py
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
from sklearn.mixture import GaussianMixture

import scipy.signal

from scipy.signal import butter, filtfilt, freqz

from eryn.moves import GaussianMove, StretchMove
from eryn.utils import TransformContainer, SearchConvergeStopping, Stopping
from eryn.state import BranchSupplemental

sys.path.append("../..")
from utility_funcs.glitch_code.glitch_shapelet_analytical_waveform import combine_shapelet_link12_frequency_domain, tdi_shapelet_link12_frequency_domain
from utility_funcs.sampling_funcs.group_stretch_proposal import MeanGaussianGroupMove as group_stretch, SelectedCovarianceGroupMove

from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity


from bbhx.utils.constants import *
from bbhx.utils.transform import *
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase

## set the GPU to use
GPU_AVAILABLE = False
if GPU_AVAILABLE:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(0)
    GPU_AVAILABLE = True
else:
    xp = np

branch_names = ["glitch","noise"]

### setting the dimensionality of the parameter to fix
ndims = {"glitch": 3, "noise": 3}

### setting the Reversible Jump on the glitches allowing between 1 and 0 glitches
nleaves_max = {"glitch": 1, "noise": 1}
nleaves_min = {"glitch": 0, "noise": 1} 

### setting the number of walkers to use and the number of temperatures 

nwalkers = 25
nfriends = nwalkers
ntemps = 10
Tmax = np.inf
tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax) # here the maximum temperature is the to infinite so that we ensure sampling the priors ( see https://arxiv.org/abs/2303.02164 )

##  setup branch supplemental for glitches to carry group stretch information
closest_inds = -np.ones((ntemps, nwalkers, nleaves_max["glitch"], nfriends), dtype=int)
closest_inds_cov = -np.ones((ntemps, nwalkers, nleaves_max["glitch"]), dtype=int)

branch_supps = {
    "glitch": BranchSupplemental(
        {"inds_closest": closest_inds, "inds_closest_cov": closest_inds_cov}, base_shape=(ntemps, nwalkers, nleaves_max["glitch"])
    ),
    "mbh": None
}

## -------------------   uploading Spritz data set ------------------ ##

# Open the HDF5 file in read mode
with h5py.File('../LDC2_Spritz_only_GW.h5', 'r') as f:
    # Access the dataset named 'time_series'
    data = f['time_series'][:]
    
    # Split the data into time and TDI channel X
    time = data[500:, 0] - data[500:, 0][0] # First column (time) #NB I am shifting the time to start at 0
    data_tdi_X = -data[500:, 1]  # Second column (TDI channel X)
    data_tdi_Y = -data[500:, 2]  # Third column (TDI channel Y)
    data_tdi_Z = -data[500:, 3]  # Forth column (TDI channel Z)

dt = time[1]-time[0]
freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt

#  ----- Generating the MBHBs from https://mikekatz04.github.io/BBHx/html/bbhx_tutorial.html-------- #

f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2 # phase at f_ref
m1 = 1323277.47932  #/(1 + 1.73941)
m2 =  612485.50602999  #/(1 + 1.73941)
M = (m1 + m2) 
q = m2 / m1  # m2 less than m1 
a1 = 0.747377 # spin 1
a2 =  0.8388   # spin 2
dist = 36.90249521628649 # luminosity distance
inc = np.pi/3 #inclination
beta = -0.30300442294174235  # ecliptic latitude
lam =   1.2925183861048521 # ecliptic longitude
psi = np.pi/6 # polarization angle
t_ref = 2627744.9218792617

if GPU_AVAILABLE:   
    force_backend = "cuda12x"
else:
    force_backend = "cpu"
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=False),
    response_kwargs=dict(TDItag="AET"),
    force_backend=force_backend)

fill_dict = {"ndim_full": 12,
    "fill_values": np.array([0.0]),
    "fill_inds": np.array([6]),}


 #  "fill_values": np.array([np.log(M),q,a1,a2,0.0]), # 
 #   "fill_inds": np.array([0,1,2,3,6]),

## these are the parameters to estimate

mbh_injection_params = np.array([
    M, 
    q,
    a1, 
    a2,
    dist,
    phi_ref,
    inc,
    lam,
    beta,
    psi,
    t_ref,]) 

mbh_injection_params[0] = np.log(mbh_injection_params[0])  # Takes the logarithm of the mass of the primary black hole.
mbh_injection_params[6] = np.cos(mbh_injection_params[6])  # Takes the cosine of the inclination angle.
mbh_injection_params[8] = np.sin(mbh_injection_params[8])  # Takes the sine of the ecliptic latitude 


# transforms from PE to waveform generation
parameter_transforms = {
    0: np.exp,
    4: lambda x: x * PC_SI * 1e9,  # Gpc  
    7: np.arccos,
    9: np.arcsin,
    (0, 1): mT_q,
    (11, 8, 9, 10): LISA_to_SSB,
}

transform_fn = TransformContainer(
    parameter_transforms=parameter_transforms,
    fill_dict=fill_dict,
)

# sampler treats periodic variables by wrapping them properly
periodic = { "mbh": {5: 2 * np.pi, 7: np.pi, 8: np.pi}}

# -------- creating the waveform -------- ##

bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024)

injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)

### frequency domain data ###

data_mbh_AET = wave_gen(*injection_in, **bbh_kwargs)[0]

###  time domain data in TDI A,E,T ###
data_channels_AET_TD = np.fft.irfft(data_mbh_AET,axis=-1).squeeze()


A_data,E_data, T_data = AET(data_tdi_X, data_tdi_Y, data_tdi_Z)


fft_data_tdi_A =(np.fft.rfft(A_data) * dt)[1:] # TD glitch
freqs = np.fft.rfftfreq(len(A_data), dt)[1:]  # fs =1/dt
##### ----------------  MBHBs waveform creation -------- #

def AziPolAngleL2PsiIncl(bet, lam, theL, phiL):
    """
    Convert Polar and Azimuthal angles of zS (typically orbital angular momentum L)
    to polarisation and inclination (see doc)
    @param bet is the ecliptic latitude of the source in sky [rad]
    @param lam is the ecliptic longitude of the source in sky [rad]
    @param theL is the polar angle of zS [rad]
    @param phiL is the azimuthal angle of zS [rad]
    @return polarisation and inclination
    """
    #inc = np.arccos( np.cos(theL)*np.sin(bet) + np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    #up_psi = np.cos(theL)*np.cos(bet) - np.sin(bet)*np.sin(theL)*np.cos(lam - phiL)
    #down_psi = np.sin(theL)*np.sin(lam - phiL)
    #psi = np.arctan2(up_psi, down_psi)
    inc = np.arccos( - np.cos(theL)*np.sin(bet) - np.cos(bet)*np.sin(theL)*np.cos(lam - phiL) )
    down_psi = np.sin(theL)*np.sin(lam - phiL)
    up_psi = -np.sin(bet)*np.sin(theL)*np.cos(lam - phiL) + np.cos(theL)*np.cos(bet)
    psi = np.arctan2(up_psi, down_psi)
    return psi, inc

f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2201968860015653 # phase at f_ref
m1 = 1323277.47932#*(1 + 1.73941)
m2 =  612485.50602999#*(1 + 1.73941)
M = (m1 + m2)
q = m2 / m1  # m2 less than m1 
a1 = 0.747377
a2 =  0.8388  # a1 >a2
dist =13449.011 * PC_SI * 1e6  #  #
beta = -0.30300442294174235  # ecliptic latitude
lam =   1.2925183861048521 # ecliptic longitude
inc = AziPolAngleL2PsiIncl(beta, lam, 2.691982450032945, 1.808398497592109)[1]
psi = AziPolAngleL2PsiIncl(beta, lam,  2.691982450032945,1.808398497592109)[0]
t_ref = 11526944.92187962 



### ---------  choose the frequency range --------- ###
one_year = 86400*365.26

t0= time[0]

n = int(len(A_data)/ dt)
data_freqs = np.fft.rfftfreq(n, dt)[700:] # remove DC

# frequencies to interpolate to
modes = [(2,2)]
waveform_kwargs = dict(modes=modes, direct=False, fill=True, squeeze=True, length=1024)
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=True))
data_channels = wave_gen(m1, m2, a1, a2,
                          dist, phi_ref, f_ref, inc, lam,
                          beta, psi, t_ref, freqs=freqs,
                          **waveform_kwargs)[0]

plt.figure()  # Set figure size
# Plot MBHBs and glitches with clear labels and colors
plt.loglog(freqs, np.abs(data_channels[0,:])**2, 'r', label="model (with BBHx) MBHB TDI A", alpha=1, linewidth=1.2)
plt.loglog(freqs, np.abs(fft_data_tdi_A)**2, 'b', label="DATA MBHB TDI A", alpha=1, linewidth=1.2)
plt.ylabel("TDI X")
plt.xlabel("Time [s]")
plt.ylim([(1e-20)**2,(1e-16)**2])
plt.xlim([1e-4,1e-1])
plt.legend()
plt.show()
plt.close()
breakpoint()
# time domain 
data_channels_AET_TD = np.fft.irfft(data_channels, axis=1)

##  --- time domain data to use for the analysis ------  ##

model_A = data_channels_AET_TD[0]
model_E = data_channels_AET_TD[1]
model_T =  data_channels_AET_TD[2]

plt.figure()  # Set figure size
# Plot MBHBs and glitches with clear labels and colors
#plt.plot( model_A, 'r', label="model (with BBHx) MBHB TDI X", alpha=1, linewidth=1.2)
plt.plot(data_tdi_X, 'g--', label="data Spritz MBH TDI X", alpha=1)
plt.ylabel("TDI X")
plt.xlabel("Time [s]")

plt.legend(loc="upper left")
plt.savefig("time_domain_spritz_data.pdf")
plt.show()
plt.close()
breakpoint()
