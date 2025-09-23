# ---------  Parameter estimation of the Ligh-Spritz data -------- #

import numpy as np
import os
import matplotlib.pyplot as plt
import corner
import cupy as cp
import numpy as np
from chainconsumer import ChainConsumer
import scipy.stats as stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy import interpolate
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

from scipy.signal import butter, filtfilt, freqz


from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.prior import ProbDistContainer, uniform_dist
from lisatools.Gaussian_prior import gaussian_dist
from eryn.utils import TransformContainer, SearchConvergeStopping, Stopping
from eryn.backends import HDFBackend

from eryn.moves import GaussianMove, StretchMove, GroupStretchMove , GroupMove, ReversibleJumpMove,DistributionGenerateRJ,MTDistGenMoveRJ, MTDistGenMove
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.group_stretch_proposal import MeanGaussianGroupMove_MBH

from lisatools.glitch_shapelet_analytical_waveform import combine_shapelet_link12_frequency_domain, tdi_shapelet_link12_frequency_domain
from lisatools.group_stretch_proposal import MeanGaussianGroupMove as group_stretch
from lisatools.group_stretch_proposal import SelectedCovarianceGroupMove
from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.constants import *
from bbhx.utils.transform import *

from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse

from synthetic_noise_generator import get_sinthetic_noise, get_sinthetic_psd

import matplotlib.pyplot as plt

# set random seed
np.random.seed(10)

## set the GPU to use

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(0)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using 
use_gpu = True

if use_gpu is False:
    xp = np

import corner

## Setting branches, ndims and leaves 

branch_names = ["glitch","noise","mbh"]
ndims = {"glitch": 3, "noise": 3,"mbh":11}
nleaves_max = {"glitch": 6, "noise": 1,"mbh":1}
nleaves_min = {"glitch": 3, "noise": 1,"mbh":1}

def set_figsize(column='single', ratio=None, scale=1):
    """Return figure size in inches for single or double column width"""
    golden_ratio = (5**0.5 - 1) / 2  # ~0.618
    if ratio is None:
        ratio = golden_ratio
    
    widths = {
        'single': 3.375,
        'double': 6.875
    }
    width = scale * widths[column]
    height = width * ratio
    return (width, height)

# Load base style

import matplotlib.pyplot as plt
plt.style.use('revtex_base.mplstyle')

nwalkers = 30
ntemps = 25
# I have added Tmax
Tmax = np.inf
# fill kwargs dictionary
tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax)

## definition of the likelihood function 
def log_like_fn(x_param, groups,bbh_transform_fn, data, df, freqs,time,dt, filter_tf):   
 
    glitch_params_all, beta_params_all,mbh_par =  x_param

    group_glitch, group_beta, group_mbh  = groups

    ngroups = int(group_beta.max()) + 1

    group_glitch_dict = {}
   
    for i in range(ngroups):
        index = np.where(group_glitch == i)
        if len(index) > 0:
                group_glitch_dict[i] = glitch_params_all[index]
        else:
            group_glitch_dict[i] = xp.zeros((1, 3))
   
    logl_all = []
    
    for group in range(ngroups):  
        
        shapelet_params = group_glitch_dict[group]
     
        beta_params = beta_params_all[group]

        mbh_params = mbh_par[group]

        psd_estimated = noise_models_spritz(freqs, isi_rfi_back_oms_noise_level =beta_params[0],   tmi_oms_back_level =beta_params[1],acc_level =beta_params[2], T = 8.322688660167833)

        tot_psd =  xp.asarray([psd_estimated* np.abs(filter_tf)**4,  psd_estimated* np.abs(filter_tf)**4]) ## to account for the filter

        templateXYZ = combine_shapelet_link12_frequency_domain(freqs,shapelet_params,T = 8.322688660167833,tdi_channel='second_gen').squeeze()
        
        A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2]) ## to account for the filter

        fft_template = xp.asarray([A* np.abs(filter_tf)**2, E* np.abs(filter_tf)**2]) # I have comment out T

        bbhx_waveform_gen = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=False, initial_t_val = time[0]),
        response_kwargs=dict(TDItag="AET"),   
        use_gpu=use_gpu)

        t_obs_start=time[0] 
        t_obs_end=time[-1] 

        bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=t_obs_start/one_year, t_obs_end=t_obs_end/one_year,shift_t_limits=True)
        bbh_params_in = bbh_transform_fn.both_transforms(mbh_params, return_transpose=True)
        AET_mbh =bbhx_waveform_gen(*bbh_params_in, **bbh_kwargs).squeeze()*dt
        fft_AET_mbh =  xp.asarray( [AET_mbh[0,:].get()*np.abs(filter_tf)**2, AET_mbh[1,:].get()*np.abs(filter_tf)**2])

        ## both MBHB waveform and glitches
        fft_template += fft_AET_mbh

        xp.get_default_memory_pool().free_all_blocks()
        
        plt.figure()
        plt.title('templeate_data')
        plt.loglog(freqs,np.sqrt(2*np.abs(data.get()[0])**2/dt/len(data.get()[0])),'b-',label='data')
        plt.loglog(freqs,np.sqrt(2*np.abs(fft_template.get()[0])**2/dt/len(fft_template.get()[0])),'k-',label='templeate')
        plt.loglog(freqs,np.sqrt(np.abs(tot_psd.get()[0])),'y-',label='psd')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Fractional frequency [strain]')
        plt.legend()
        plt.grid()          
        plt.savefig("templeate_data.png")
        plt.close()
  
        
        ## I compute the log likelihood here (refer to our paper for the derivation)
        logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data - fft_template)).real /(tot_psd ), axis=0).sum() )
        logl += -  xp.sum(xp.log(tot_psd), axis=0).sum()
        logl = logl[np.newaxis]
        if xp.any(xp.isnan(logl)):
            breakpoint()  
        logl_all.append(logl)
      
    logl_out = np.concatenate(logl_all)
   
    return logl_out.get()

## -------------------   uploading Spritz data set ------------------ ##
import h5py

# Open the HDF5 file in read mode
with h5py.File('LDC2_Spritz_noise_and_glitches.h5', 'r') as f:
    # Access the dataset named 'time_series'
    data = f['time_series'][:]
    
    # Split the data into time and TDI channel X (Spritz)
    time = data[500:, 0]  - data[500:, 0][0] # First column (time)
    data_tdi_X = -data[500:, 1]  # Second column (TDI channel X)
    data_tdi_Y = -data[500:, 2]  # Third column (TDI channel X)
    data_tdi_Z = -data[500:, 3]  # Forth column (TDI channel X)

dt = time[1]-time[0]

##### ----------------  MBHBs waveform creation -------- #

f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2 # phase at f_ref
m1 = 1323277.47932  #/(1 + 1.73941)
m2 =  612485.50602999  #/(1 + 1.73941)
M = (m1 + m2)
q = m2 / m1  # m2 less than m1 
a1 = 0.747377
a2 =  0.8388  # a1 >a2
dist = 36.90249521628649 
inc = np.pi/3
beta = -0.30300442294174235  # ecliptic latitude
lam =   1.2925183861048521 # ecliptic longitude
psi = np.pi/6 # polarization angle
t_ref =2627744.9218792617
time0 = 0
# wave generating class
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=False,initial_t_val = time0),
    response_kwargs=dict(TDItag="AET"),   
    use_gpu=use_gpu)

# for transforms
fill_dict = {
    "ndim_full": 12,
       "fill_values": np.array([0.0]),
    "fill_inds": np.array([6]),
}

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
    t_ref,
]) 


# get the right parameters
mbh_injection_params[0] = np.log(mbh_injection_params[0])  # Takes the logarithm of the mass of the primary black hole.
mbh_injection_params[6] = np.cos(mbh_injection_params[6])  # Takes the cosine of the inclination angle.
mbh_injection_params[8] = np.sin(mbh_injection_params[8])  # Takes the sine of the ecliptic latitude 


# transforms from pe to waveform generation
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

### ---------  choose the frequency range --------- ###

freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt

# creating the waveform ##
one_year = 86400*365.26
bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=time[0]/one_year, t_obs_end=time[-1]/one_year,shift_t_limits=True)
injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)

# frequency domain 
data_mbh_AET = wave_gen(*injection_in, **bbh_kwargs)[0]

# time domain 
data_channels_AET_TD = np.fft.irfft(data_mbh_AET,axis=-1).squeeze()

##  --- time domain data to use for the analysis ------  ##

A_data,E_data, T_data = AET(data_tdi_X, data_tdi_Y, data_tdi_Z)

A_data_tot = A_data + data_channels_AET_TD.get()[0]
E_data_tot = E_data + data_channels_AET_TD.get()[1]
T_data_tot = T_data + data_channels_AET_TD.get()[2]


## ---------------------- noise models ---------------- ##

def noise_models_spritz(f,  isi_rfi_back_oms_noise_level = np.sqrt( (6.35e-12)**2 + (3.32e-12)**2+ (3.0E-12)**2 ),    tmi_oms_back_level = np.sqrt( (1.42e-12)**2 +(3.0E-12)**2 ),  acc_level = 2.4e-15,    T = 8.322688660167833):

    c = 299792458.0

    # Common TDI factor for first gen TDI, which can be factorized as (1 - D^2) * X_0, with X_0 being a simple Michelson.
    Cxx = (np.abs(1 - np.exp(-2j*np.pi*f*T)**4)*np.abs(1 - np.exp(-2j*np.pi*f*T)**2))**2

    # conversion factors into ffd units used in LDC
    lamb = 1064.5e-9
    nu0 = c / lamb
    # conversion: divide by lambda to get cycles, take a derivative to get frequency in Hz, divide by nu0 to get ffd
    disp_2_ffd = (2 * np.pi * f / lamb / nu0)**2
    # conversion: divide by lambda to get cycles/s^2, integrate to get frequency in Hz, divide by nu0 to get ffd
    acc_2_ffd = (1 / (lamb * 2 * np.pi * f ) / nu0)**2

    # Backlink noises. This one has a low-freq relaxation.
    backlink_noise_level = 3.0E-12
    backlink_noise_lowf = backlink_noise_level**2 * (2e-3 / f)**4

    # these are the transfer functions
    # Note that ISI, RFI OMS and backlink have the same one. 
    isi_rfi_readout_transfer_AA = 2 * Cxx * (2 + np.cos(2 * np.pi * f *T))
    tmi_readout_transfer_AA = Cxx * (3 + 2 * np.cos(2 * np.pi * f *T) + np.cos(4 * np.pi * f *T))

    # these are the noises in TDI
    isi_rfi_bl_oms_AA_ffd = isi_rfi_readout_transfer_AA * disp_2_ffd * isi_rfi_back_oms_noise_level**2
    tmi_bl_oms_AA_ffd = tmi_readout_transfer_AA * disp_2_ffd * tmi_oms_back_level**2

    # Tranfser functions into RFI and TMI are identical to the corresponding readout noise terms (there are two instances of this noise per OB)
    rfi_backlink_lowf_AA_ffd = isi_rfi_readout_transfer_AA * disp_2_ffd * backlink_noise_lowf
    tmi_backlink_lowf_AA_ffd = tmi_readout_transfer_AA * disp_2_ffd * backlink_noise_lowf

    # TM noises. It's implemented with a single knee frequency at 0.4e-3 Hz.
    tm_noise = acc_level**2 * (1 + (0.4e-3/f)**2)

    # these are the transfer functions
    TM_transfer_AA = 4 * tmi_readout_transfer_AA

    # these are the noise terms
    tm_noise_AA_ffd = TM_transfer_AA * tm_noise * acc_2_ffd

    total_noise_AA = tm_noise_AA_ffd + isi_rfi_bl_oms_AA_ffd + tmi_bl_oms_AA_ffd + rfi_backlink_lowf_AA_ffd + tmi_backlink_lowf_AA_ffd
 
    return total_noise_AA

plots = False # Note that if you run this if then you cannot keep evaluating the code but you need to set plots=False and then run again the rest of the code
if plots == True:

    with h5py.File('LDC2_Spritz_glitch.h5', 'r') as f:
        # Access the dataset named 'time_series'
        data = f['time_series'][:]
        
        # Split the data into time and TDI channel X
        time = data[500:, 0]  - data[500:, 0][0] # First column (time)
        data_tdi_glitch_X = -data[500:, 1]  # Second column (TDI channel X)
        data_tdi_glitch_Y = -data[500:, 2]  # Third column (TDI channel X)
        data_tdi_glitch_Z = -data[500:, 3]  # Forth column (TDI channel X)

    A_data_glitch,E_data_glitch, T_data_glitch = AET(data_tdi_glitch_X, data_tdi_glitch_Y, data_tdi_glitch_Z)

    A_glitch_fft = np.fft.rfft(A_data_glitch) * dt # TD glitch
    A_total_fft = np.fft.rfft(A_data_tot) * dt # TD glitch
    freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt

    plt.figure(figsize=set_figsize('single', ratio =1))  # Set figure size
    # Plot MBHBs and glitches with clear labels and colors
    #plt.plot(time, A_data_tot, 'r', label="Glitches, noise, and MBHB", alpha=1, linewidth=1.2)
    plt.plot(time, A_data_glitch, 'g', label="Glitches signals only", alpha=1)
    #plt.plot(time, data_channels_AET_TD.get()[0], 'b', label="MBHB signal only", alpha=1, linewidth=2)
    # Labels with proper formatting
    plt.ylabel("TDI A [Hz/$\sqrt{\mathrm{Hz}}$]")
    plt.xlabel("Time [s]")
    # Enable grid with subtle dashed lines

    # Place the legend outside the plot for better clarity
    plt.legend(loc="upper left")
    plt.xlim([time[0],time[-1]])
    plt.savefig("time_domain_spritz_data.pdf")
    plt.show()
    plt.close()

    plt.figure(figsize=set_figsize('single', ratio =0.9))  # Set figure size
    # Plot MBHBs and glitches with clear labels and colors
    plt.loglog(freqs, np.sqrt(2*np.abs(A_total_fft)**2/dt/len(A_data_tot)) , 'r', label="Glitches, noise, and MBHB", alpha=1)
    plt.loglog(freqs, np.sqrt(2*np.abs(A_glitch_fft)**2/dt/len(A_data_glitch)), 'g--', label="Glitches signals only", alpha=1)
    plt.loglog(freqs,  np.sqrt(2*np.abs(data_mbh_AET.get()[0])**2/dt/len(data_channels_AET_TD.get()[0])), 'b', label="MBHB signal only", alpha=1)
    plt.loglog(freqs, np.sqrt(noise_models_spritz(freqs)), 'k--', label="PSD noise model", alpha=1, linewidth=2)
    #plt.loglog(freqs, np.sqrt(noise_models_spritz(freqs) + 2*np.abs(A_glitch_fft)**2/dt/len(A_data_glitch) + 2*np.abs(A_total_fft)**2/dt/len(A_data_tot)), 'y:', label="PSD of the sum", alpha=0.5, linewidth=4)
    # Labels with proper formatting
    plt.grid()
    plt.ylabel("TDI A [1/$\sqrt{\mathrm{Hz}}$]")
    plt.xlabel("Frequency [Hz]")
    # Enable grid with subtle dashed lines
    # Place the legend outside the plot for better clarity
    plt.legend(loc="best", frameon=True)
    plt.ylim([5e-25,3*1e-19])
    plt.xlim([2.5e-5,1e-1])
    plt.savefig("frequency_domain_spritz_data.png")
    plt.show()
    plt.close()
    breakpoint()

    ########## -----------------  #########

    # Compute spectrogram
    nperseg = int(len(A_data_tot) // 100)  # Adjust for resolution
    fxx, txx, Sxx = scipy.signal.spectrogram(A_data_tot, fs=1.0/dt, nperseg=nperseg, scaling='density')

    # Plot
    t_start = time[0]  # Assuming 'time' is your original time array
    t_end = time[-1]
    txx_corrected = np.linspace(t_start, t_end, len(txx))

    # Convert power to dB
    Sxx_dB = 10 * np.log10(Sxx)

    # Plot
    plt.figure(figsize=set_figsize('single', ratio =0.5))

    pcm = plt.pcolormesh(txx_corrected, fxx, Sxx_dB, shading="gouraud", cmap="inferno")


    # Color bar adjustments
    cbar = plt.colorbar(pcm)
    cbar.set_label("Power/Frequency [dB/Hz]")

    # Axis labels
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency [Hz]")

    # Adjust ticks
    plt.tick_params(axis="both")

    # Log scale for clarity
    plt.yscale("log")
    plt.ylim([1e-4, 1e-1])  # Focus on relevant frequencies


    # Save high-quality output
    plt.savefig("time_frequency_spritz.png")

    # Show the plot
    plt.show()
    plt.close()

## -----------  applied filter to the data --------   ##

# Define filter parameters
sampling_rate = 1/dt  # Hz
nyquist_freq = sampling_rate / 2
cutoff_freq = 1e-3  # Hz ok if only noise ## was 1e-2 , I had to use a different one
normalized_cutoff = cutoff_freq / nyquist_freq

# Design a first-order Butterworth filter
b, a = butter(N=1, Wn=normalized_cutoff, btype='low', analog=False)

# Example: Apply the filter to your time series data
A_data_filtered = filtfilt(b, a, A_data_tot)[500:-500]
E_data_filtered = filtfilt(b, a, E_data_tot)[500:-500]

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(time, A_data_tot, 'r', label="Original Signal" , alpha=0.7)
plt.plot(time[500:-500], A_data_filtered,'b',label="Filtered Signal" , linewidth=2)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.title("Time-Domain Signal: Original vs. Filtered")
plt.savefig("filtered_signal.png")
plt.legend()
plt.grid()
plt.show()

##  ------------- frequency data ----------------  ##

# we need to redefine the frequency axes since we filter the data 
time = time[500:-500]

freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt

Anfft = np.fft.rfft(A_data_filtered ) * dt # TD glitch
Enfft = np.fft.rfft(E_data_filtered) * dt # TD glitch

fft_data_AE = xp.array([Anfft,Enfft])  # I have comment out T


## ---------------------- noise models ---------------- ##

def noise_models_spritz(f,  isi_rfi_back_oms_noise_level = np.sqrt( (6.35e-12)**2 + (3.32e-12)**2+ (3.0E-12)**2 ),    tmi_oms_back_level = np.sqrt( (1.42e-12)**2 +(3.0E-12)**2 ),  acc_level = 2.4e-15,    T = 8.322688660167833):

    c = 299792458.0

    # Common TDI factor for first gen TDI, which can be factorized as (1 - D^2) * X_0, with X_0 being a simple Michelson.
    Cxx = (np.abs(1 - np.exp(-2j*np.pi*f*T)**4)*np.abs(1 - np.exp(-2j*np.pi*f*T)**2))**2

    # conversion factors into ffd units used in LDC
    lamb = 1064.5e-9
    nu0 = c / lamb
    # conversion: divide by lambda to get cycles, take a derivative to get frequency in Hz, divide by nu0 to get ffd
    disp_2_ffd = (2 * np.pi * f / lamb / nu0)**2
    # conversion: divide by lambda to get cycles/s^2, integrate to get frequency in Hz, divide by nu0 to get ffd
    acc_2_ffd = (1 / (lamb * 2 * np.pi * f ) / nu0)**2

    # Backlink noises. This one has a low-freq relaxation.
    backlink_noise_level = 3.0E-12
    backlink_noise_lowf = backlink_noise_level**2 * (2e-3 / f)**4

    # these are the transfer functions
    # Note that ISI, RFI OMS and backlink have the same one. 

    # these are the transfer functions
    # Note that ISI, RFI OMS and backlink have the same one. 
    isi_rfi_readout_transfer_AA = 2 * Cxx * (2 + np.cos(2 * np.pi * f *T))
    tmi_readout_transfer_AA = Cxx * (3 + 2 * np.cos(2 * np.pi * f *T) + np.cos(4 * np.pi * f *T))

    # these are the noises in TDI
    isi_rfi_bl_oms_AA_ffd = isi_rfi_readout_transfer_AA * disp_2_ffd * isi_rfi_back_oms_noise_level**2
    tmi_bl_oms_AA_ffd = tmi_readout_transfer_AA * disp_2_ffd * tmi_oms_back_level**2

    # Tranfser functions into RFI and TMI are identical to the corresponding readout noise terms (there are two instances of this noise per OB)
    rfi_backlink_lowf_AA_ffd = isi_rfi_readout_transfer_AA * disp_2_ffd * backlink_noise_lowf
    tmi_backlink_lowf_AA_ffd = tmi_readout_transfer_AA * disp_2_ffd * backlink_noise_lowf

    # TM noises. It's implemented with a single knee frequency at 0.4e-3 Hz.
    tm_noise = acc_level**2 * (1 + (0.4e-3/f)**2)

    # these are the transfer functions
    TM_transfer_AA = 4 * tmi_readout_transfer_AA

    # these are the noise terms
    tm_noise_AA_ffd = TM_transfer_AA * tm_noise * acc_2_ffd

    total_noise_AA = tm_noise_AA_ffd + isi_rfi_bl_oms_AA_ffd + tmi_bl_oms_AA_ffd + rfi_backlink_lowf_AA_ffd + tmi_backlink_lowf_AA_ffd
 
    return total_noise_AA


## --------- Compute the unfiltered noise PSD ---------- ##

df = freqs[1] - freqs[0]  # 1 / (dt * len(t_in))

fmin = 2e-5
fmax = 2.5e-2
frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them

freqs_cut =  np.array(freqs[frequencymask])

## --------------------------  Get the filter's frequency response -------------  ##
_, h = freqz(b, a, worN=len(freqs), fs=1/dt)

## ------------- Apply the filter in the frequency domain  ------------ ##

Sa_unfiltered = noise_models_spritz(freqs)   
Se_unfiltered =  noise_models_spritz(freqs)   

Sa_filtered = Sa_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response
Se_filtered = Se_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response

h = h[frequencymask]

fft_data_cutted = xp.array([fft_data_AE[0,:][frequencymask],fft_data_AE[1,:][frequencymask] ])

## ------- visualizing the data in frequency and time domain ------ ##

from scipy.signal import welch

freq_welch, psd_data_E = welch(E_data, fs=1/dt, window=('kaiser',15), nperseg=len(E_data)//2, noverlap=50)
freq_welch, psd_data_A = welch(A_data, fs=1/dt, window=('kaiser',15), nperseg=len(E_data)//2, noverlap=50)
freq_welch_f, psd_data_A_filtered= welch(A_data_filtered, fs=1/dt, window='boxcar', nperseg=len(A_data_filtered)//2, noverlap=50)
freq_welch_f, psd_data_E_filtered= welch(E_data_filtered, fs=1/dt, window='boxcar', nperseg=len(E_data_filtered)/2, noverlap=50)
plt.figure()
plt.title('TDI A: ')

plt.loglog(freq_welch_f[1:],np.sqrt( psd_data_A_filtered)[1:],'b',label='psd data A')
plt.loglog(freq_welch_f[1:],np.sqrt(psd_data_E_filtered)[1:],'y',label='psd data E ')
#plt.loglog(frequencies[4:],np.sqrt(psd_data_A)[4:],'m--',label='psd data welch A')
#plt.loglog(frequencies[4:],np.sqrt(psd_data_E)[4:],'b-',label='psd data welch E')
plt.loglog(freq_welch[1:],np.sqrt(psd_data_E)[1:],'r',label='psd data welch E')
plt.loglog(freq_welch[1:],np.sqrt(psd_data_A)[1:],'m',label='psd data welch A')
plt.loglog(freqs[1:],np.sqrt(Sa_filtered)[1:],'k--',label='model PSD E filtered')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Fractional frequency [strain]')
#plt.ylim([1e-23,1e-20])
#plt.xlim([1e-4,1e-2])
plt.legend()
plt.grid()          
plt.savefig("spritz_data_psd_vs_model.png")
plt.close()


if __name__ == "__main__":
    
    ##### glitch priors ######

    priors_in = {
    0: uniform_dist(432000,2627995), 
    1: uniform_dist(-32,-20), 
    2: uniform_dist(1, 1e4)} 

    priors = {}
    priors["glitch"] = ProbDistContainer(priors_in)

    ##### noise priors ######

    priors_noise = {
    0: uniform_dist((7.5e-12),(8e-12)),
    1: uniform_dist((2.5e-12),(3.5e-12)),
    2: uniform_dist((2.2e-15),(2.7e-15)),}

    priors['noise'] = ProbDistContainer(priors_noise) 

    # I am now defining the covariance matrix for the gaussian move in the in -model move # 
    nfriends = nwalkers

    gibbs = []
    for i in range(nleaves_max["glitch"]):
        tmp = np.zeros((nleaves_max["glitch"], ndims["glitch"]), dtype=bool)
        tmp[i] = True

        gibbs.append(("glitch", tmp))
   
    gs = group_stretch(nfriends=nfriends,gibbs_sampling_setup=gibbs,  n_iter_update=100,live_dangerously=True) #
    scgm = SelectedCovarianceGroupMove(nfriends=1, gibbs_sampling_setup=gibbs,n_iter_update=100,live_dangerously=True)
        

    priors["mbh"] = ProbDistContainer(
        {
            0: uniform_dist(np.log(1e5), np.log(1e8)),
            1: uniform_dist(0.01, 0.999999999),
            2: uniform_dist(-0.99999999, +0.99999999),
            3: uniform_dist(-0.99999999, +0.99999999),
            4: uniform_dist(0.01, 1000.0),
            5: uniform_dist(0.0, 2 * np.pi),
            6: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
            7: uniform_dist(0.0, 2 * np.pi),
            8: uniform_dist(-1.0 + 1e-6, 1.0 - 1e-6),
            9: uniform_dist(0.0, np.pi),
            10: uniform_dist(2.627744e6-24,2.627744e6+24),
        }
    ) 

    #### Notice that the covariances for the glitches and MBHB here are estimated from the search pipeline ####

    cov1 = { "glitch":np.array([[ 0.07532411,  0.00345846, -0.03929037],
                        [ 0.00345846,  0.00017702, -0.00198148],
                        [-0.03929037, -0.00198148,  0.02253148]])}

    cov2 = {  "glitch":  np.array([[ 0.07889849,  0.00363467, -0.0417983 ],
         [ 0.00363467,  0.00018766, -0.0021186 ],
         [-0.0417983 , -0.0021186 ,  0.02435929]])}


    cov3 = { "glitch": np.array([[ 1.17074335e-01,  1.24757036e-03, -5.10152516e-02],
         [ 1.24757036e-03,  2.51816742e-05, -7.40131379e-04],
         [-5.10152516e-02, -7.40131379e-04,  2.98566204e-02]])}
    
    gibbs_sampling_mbh = [
        ("mbh", np.zeros((nleaves_max["mbh"], ndims["mbh"]), dtype=bool))
        for _ in range(nleaves_max["mbh"])]

    for i in range(nleaves_max["mbh"]):
        gibbs_sampling_mbh[i][-1][i] = True
  

    cov_noise = { 
    "noise": np.diag(np.ones(ndims['noise'])) * 1e-35  }

    cov_mbh = {"mbh":np.array([[ 7.59759363e-09, -9.27081301e-09,  1.87886796e-08,
        -2.31910783e-08,  3.29312410e-06, -1.43918451e-07,
         1.84765584e-08, -8.91942228e-09, -2.92788051e-08,
        -9.24775317e-08,  1.56538542e-06],
       [-9.27081301e-09,  2.32967671e-08, -1.37599117e-08,
         4.22964559e-08, -6.40325232e-06,  2.30905724e-08,
        -6.10007948e-08, -5.03176451e-11,  4.88251009e-08,
         1.46677747e-07,  3.17061440e-06],
       [ 1.87886796e-08, -1.37599117e-08,  7.43992195e-08,
        -8.40131216e-08,  1.24817749e-05, -9.11682067e-07,
         2.11548667e-08, -7.26079187e-08, -1.39404590e-07,
        -4.53365796e-07,  1.23197494e-05],
       [-2.31910783e-08,  4.22964559e-08, -8.40131216e-08,
         1.71565640e-07, -2.80736941e-05,  1.12549816e-06,
        -1.35202518e-07,  1.01733521e-07,  2.83041036e-07,
         8.97036410e-07, -5.12377685e-06],
       [ 3.29312410e-06, -6.40325232e-06,  1.24817749e-05,
        -2.80736941e-05,  7.29897750e-03, -1.85725414e-04,
         6.09378178e-05, -3.48563256e-06, -5.71473536e-05,
        -1.75115302e-04,  4.94251060e-04],
       [-1.43918451e-07,  2.30905724e-08, -9.11682067e-07,
         1.12549816e-06, -1.85725414e-04,  1.45323705e-05,
        -7.63721626e-08,  1.29093220e-06,  2.25569610e-06,
         7.38171210e-06, -1.83089537e-04],
       [ 1.84765584e-08, -6.10007948e-08,  2.11548667e-08,
        -1.35202518e-07,  6.09378178e-05, -7.63721626e-08,
         8.03816410e-07,  2.14030640e-07, -3.04775803e-07,
        -8.79066706e-07, -1.52137892e-05],
       [-8.91942228e-09, -5.03176451e-11, -7.26079187e-08,
         1.01733521e-07, -3.48563256e-06,  1.29093220e-06,
         2.14030640e-07,  2.25311672e-07,  1.73775000e-07,
         5.75900078e-07, -1.48797004e-05],
       [-2.92788051e-08,  4.88251009e-08, -1.39404590e-07,
         2.83041036e-07, -5.71473536e-05,  2.25569610e-06,
        -3.04775803e-07,  1.73775000e-07,  5.60084854e-07,
         1.72733318e-06, -1.27821828e-05],
       [-9.24775317e-08,  1.46677747e-07, -4.53365796e-07,
         8.97036410e-07, -1.75115302e-04,  7.38171210e-06,
        -8.79066706e-07,  5.75900078e-07,  1.72733318e-06,
         5.51136412e-06, -4.52671970e-05],
       [ 1.56538542e-06,  3.17061440e-06,  1.23197494e-05,
        -5.12377685e-06,  4.94251060e-04, -1.83089537e-04,
        -1.52137892e-05, -1.48797004e-05, -1.27821828e-05,
        -4.52671970e-05,  3.75076174e-03]])}

    moves = [
     (gs, 0.5),
      (scgm, 0.5),
      (StretchMove(gibbs_sampling_setup ="noise",live_dangerously=True),0.85),
      (GaussianMove(cov_noise,gibbs_sampling_setup ="noise"),0.15),
      (SkyMove(which="both",gibbs_sampling_setup=gibbs_sampling_mbh), 0.4),
    (SkyMove(which="long",gibbs_sampling_setup=gibbs_sampling_mbh), 0.3),
    (SkyMove(which="lat",gibbs_sampling_setup=gibbs_sampling_mbh), 0.3),
    (StretchMove(gibbs_sampling_setup=gibbs_sampling_mbh,live_dangerously=True),0.85),
    (GaussianMove(cov_mbh,gibbs_sampling_setup=gibbs_sampling_mbh),0.05)] 
    
    # make sure to start near the proper setup 
    inds = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"]), dtype=bool),
            "noise": np.ones((ntemps, nwalkers, nleaves_max["noise"]),  dtype=bool),
            "mbh": np.ones((ntemps, nwalkers, nleaves_max["mbh"]),  dtype=bool)
        }

    for nwalk in range(nwalkers):
        for temp in range(ntemps): 
            nl = np.random.randint(nleaves_min["glitch"], nleaves_max["glitch"]+1) 
            inds_tw=np.zeros(nleaves_max["glitch"],dtype=bool) 
            inds_tw[:nl]=True
            np.random.shuffle(inds_tw)
            inds['glitch'][temp,nwalk]=inds_tw

    ### ---------  coordinates for the glitches mbh and noise --------- ###
    compute_coords = True
    if compute_coords ==True:
        print('computing coordinates')

        fp_mbh = np.array([ 1.44755232e+01,  4.62717697e-01,  7.46099636e-01,  8.38969922e-01,
        3.67607670e+01,  1.21072503e+00,  4.99152665e-01,  1.29277615e+00,
       -2.97581521e-01,  5.25193280e-01,  2.62774464e+06])
        ## GMM over glitches since they are gaussian 
        x_glitch =np.array([[5.61800080e+05, -2.75800601e+01,  1.04721727e+01],
                        [ 2.15999998e+06, -2.75849914e+01,  1.05039976e+01],
                        [ 2.54075603e+06, -2.86751502e+01,  2.82044064e+01]])

        ### ------------- #######

        factor = 1e-9
        ndim_tot = ndims['glitch'] +ndims['mbh']
        cov = np.ones(ndim_tot ) * 1e-3
        cov[9] = 1e-7
        cov[-1] = 1e-7

        start_like = np.zeros((nwalkers * ntemps))
            
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers, ndim_tot+2*ndims['glitch'] ))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
            
        tmp[fix, :ndims['glitch']] = (x_glitch[0]* (1. + factor * cov[:ndims['glitch']] * np.random.randn(nwalkers * ntemps,ndims['glitch'])))[fix]
        tmp[fix, ndims['glitch']:2*ndims['glitch']] = (x_glitch[1]* (1. + factor * cov[:ndims['glitch']] * np.random.randn(nwalkers * ntemps,ndims['glitch'])))[fix]
        tmp[fix,2*ndims['glitch'] :3*ndims['glitch']] = (x_glitch[2]* (1. + factor * cov[:ndims['glitch']] * np.random.randn(nwalkers * ntemps,ndims['glitch'])))[fix]
        tmp[fix, 3*ndims['glitch']:] = (mbh_injection_params[None, :] * (1. + factor * cov[ndims['glitch']:] * np.random.randn(nwalkers * ntemps,ndims['mbh'])))[fix]

        # Apply modulo operations to the MBH parameters only
        tmp[:, 14] = tmp[:, 14] % (2 * np.pi)
        tmp[:, 16] = tmp[:, 16] % (2 * np.pi)
        tmp[:, 18] = tmp[:, 18] % (1 * np.pi)

        logp = priors["glitch"].logpdf(tmp[:, :3*ndims['glitch']]) + priors["mbh"].logpdf(tmp[:, 3*ndims['glitch']:])

        fix = np.isinf(logp)
        if np.all(fix):
            breakpoint()


        coords = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"], ndims["glitch"])),
            "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"])),
            "mbh": np.zeros((ntemps, nwalkers, nleaves_max["mbh"], ndims["mbh"]))}

    
        for nn in range(nleaves_max["glitch"]-3):

            coords["glitch"][:, :, nn] =priors["glitch"].rvs(size=(ntemps, nwalkers,))

    
        coords["glitch"][:, :, nleaves_max["glitch"]-3:nleaves_max["glitch"]] = tmp[:, :3*ndims['glitch']].reshape(ntemps, nwalkers, nleaves_max["glitch"]-3,  ndims['glitch']) 
        coords["noise"] = priors["noise"].rvs(size=(ntemps, nwalkers,nleaves_max["noise"]))
        coords["mbh"] = tmp[:,  3*ndims['glitch']:].reshape(ntemps, nwalkers, nleaves_max["mbh"], ndims['mbh'])
        


        rj_moves_mix = [(DistributionGenerateRJ(priors,nleaves_max,nleaves_min,gibbs_sampling_setup='glitch'),0.95)]

        list0 = x_glitch[:,0]
        listDV = x_glitch[:,1]
        listtau = x_glitch[:,2]

        covariance_tot = np.array([cov1['glitch'],cov2['glitch'],cov3['glitch']])
        # get rid of this
        for i in range(len(list0)):
            temp_kw={}
            here_d={
                    0: gaussian_dist(list0[i],covariance_tot[i, 0, 0]),
                    1: gaussian_dist(listDV[i], covariance_tot[i, 1,1]) ,
                    2: gaussian_dist(listtau[i],covariance_tot[i, 2, 2]),    
                }
        temp_kw["glitch"]=ProbDistContainer(here_d)
        rj_moves_mix.append((DistributionGenerateRJ(temp_kw, nleaves_max, nleaves_min,gibbs_sampling_setup='glitch'),0.05) )
        


####### ------------ #######
   

def update_fn(i, res, samp):
    
        max_it_update=100
        mem =500

        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()[-mem:,0]))
        print("min last loglike",np.min(samp.get_log_like()[-mem:,0]))

        nleaves = samp.get_nleaves()['glitch'][:,0,:].reshape(-1)
        nleaves_with_3 = np.sum(nleaves == 3)/len(nleaves)
        nleaves_with_4 = np.sum(nleaves == 4)/len(nleaves)
        nleaves_with_5 = np.sum(nleaves == 5)/len(nleaves)
        nleaves_with_6 = np.sum(nleaves == 6)/len(nleaves)
        print("nleaves_frac", np.array([nleaves_with_3,nleaves_with_4,nleaves_with_5,nleaves_with_6]))
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
            print("swap \n",samp.swap_acceptance_fraction)
            print("rj \n",samp.rj_acceptance_fraction[0] )
        noise_sampler= samp.get_chain()["noise"][-mem:,0].reshape(-1,ndims['noise'])
        c = ChainConsumer()
        parameter_labels = ['$isi+rfi_OMS$','$tmi$','$TM$']
        c.add_chain(noise_sampler, parameters=parameter_labels, name='noise', color='#6495ed')
        c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
        c.add_marker([ 7.768197989237916e-12, 3.3190962625389463e-12 ,  2.4e-15], marker_style="x", marker_size=500, color='#DC143C')

        fig = c.plotter.plot(figsize=(8,8), legend=True)
        plt.savefig("noise_spritz.png", dpi=300)
        plt.close()

        likelihood = samp.get_log_like()[-mem:, 0,:]
            
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.ylim([ 1.353*1e7 +3400,  1.353*1e7 +3500])
        plt.xlabel('iter')
        plt.ylabel('Likelihood (single glitch)')
        plt.title('likelihood evolutions for all the walkers at 0 temperature with glitch')
        plt.grid(True)
        plt.show()
        plt.savefig("likelihood_spritz.png", dpi=300)
        plt.close()

        item_samp_mbh = samp.get_chain()["mbh"][-mem:,0][~np.isnan(samp.get_chain()["mbh"][-mem:,0])].reshape(-1,ndims['mbh'])
                
        c = ChainConsumer()
        # Define the parameter names with LaTeX labels
        parameter_labels = [r'$\ln M$', r'$q$', r'$a_1$', r'$a_2$', r'$d_L$', r'$\phi$', r'$\cos(\theta)$', r'$\lambda$', r'$\beta$', r'$\psi$', r'$t$']
        
        fig = corner.corner(
            item_samp_mbh,
            truths=np.array([ 1.44760121e+01,  4.62854931e-01,  7.47377000e-01,  8.38800000e-01,
                        3.69024952e+01,  1.20000000e+00,  5.00000000e-01,  1.29251839e+00,
                    -2.98389103e-01,  5.23598776e-01,  2627744.9218792617]),
            truth_color="blue",  # Explicit color for truth markers
            truth_marker="s",   # Square marker for visibility
            labels=parameter_labels,
            label_kwargs={"fontsize": 10},
            plot_contours=True,
            plot_density=False,
            plot_datapoints=False,
            color='red',
            contour_kwargs={"linewidths": 2}
        )


        # Adjust plot settings for better readability
        plt.rc('axes', labelsize=15)
        plt.rc('axes', titlesize=15)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)

        # Save the plot with high DPI for better quality
        fig.savefig("mbh_corner_plot_spritz.png", dpi=300)

        item_samp_glitch_no_nans= samp.get_chain()["glitch"][-mem:,0][~np.isnan(samp.get_chain()["glitch"][-mem:,0])].reshape(-1,3)
        item_samp_glitch_no_nans[:,2] = np.log10(item_samp_glitch_no_nans[:,2])
        
        c = ChainConsumer()
        parameter_labels = ['$t0$', '$ln(DV)$', '$log(\\tau)$']
        c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
        glitch_spritz_par = x_glitch
        glitch_spritz_par[:,1] = -29.8383
        glitch_spritz_par[:,2] = 1
        #for ii in range(len(glitch_spritz_par)):
        #    c.add_marker([glitch_spritz_par[ii][0], glitch_spritz_par[ii][1], np.log10(glitch_spritz_par[ii][2])], \
        #    parameters=parameter_labels, marker_style="p", \
        #    marker_size=100, color='#DC143C')
        c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=False, serif=True)
        fig = c.plotter.plot(figsize=(8,8), legend=True)
        plt.savefig("shapelet_corner_plot_spritz.png", dpi=300)
        plt.close()
        
        random_noise_par_oms1 = np.random.choice(noise_sampler[:,0], size=1, replace=False)
        random_noise_par_oms2 = np.random.choice(noise_sampler[:,1], size=1, replace=False)
        random_noise_par_tm = np.random.choice(noise_sampler[:,2], size=1, replace=False)

        psd_estimated =( noise_models_spritz(freqs_cut, isi_rfi_back_oms_noise_level = random_noise_par_oms1, tmi_oms_back_level = random_noise_par_oms2, acc_level = random_noise_par_tm) )*np.abs(h)**4
        '''
        plt.figure()
        plt.title('Estimated psd')
        plt.loglog(freqs[1:],np.sqrt(Sa_filtered)[1:],'b-',label='psd data A model')
        plt.loglog(freqs_cut,np.sqrt(psd_estimated),'k-',label='psd data A/E estimated')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Fractional frequency [strain]')
        plt.xlim([fmin,fmax])
        plt.legend()
        plt.grid()          
        plt.savefig("estimated_model_spritz.png", dpi=300)
        plt.close()
        '''

        ##### --------- computing SNR  ######

        random_mbh_par_1 = np.random.choice(item_samp_mbh[:,0], size=1, replace=False)
        random_mbh_par_2 = np.random.choice(item_samp_mbh[:,1], size=1, replace=False)
        random_mbh_par_3 = np.random.choice(item_samp_mbh[:,2], size=1, replace=False)
        random_mbh_par_4 = np.random.choice(item_samp_mbh[:,3], size=1, replace=False)
        random_mbh_par_5 = np.random.choice(item_samp_mbh[:,4], size=1, replace=False)
        random_mbh_par_6 = np.random.choice(item_samp_mbh[:,5], size=1, replace=False)
        random_mbh_par_7 = np.random.choice(item_samp_mbh[:,6], size=1, replace=False)
        random_mbh_par_8 = np.random.choice(item_samp_mbh[:,7], size=1, replace=False)
        random_mbh_par_9 = np.random.choice(item_samp_mbh[:,8], size=1, replace=False)
        random_mbh_par_10 = np.random.choice(item_samp_mbh[:,9], size=1, replace=False)
        random_mbh_par_11 = np.random.choice(item_samp_mbh[:,10], size=1, replace=False)
      

        mbh_injection_params= np.reshape((random_mbh_par_1,random_mbh_par_2,random_mbh_par_3,random_mbh_par_4,random_mbh_par_5,random_mbh_par_6,random_mbh_par_7,random_mbh_par_8,random_mbh_par_9,random_mbh_par_10,random_mbh_par_11),-1)
        # get injected parameters after transformation
        injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)
        
        data_mbh_AET = wave_gen(*injection_in, freqs=xp.asarray(freqs_cut), direct=False, fill=True, squeeze=True, length=1024).squeeze()

        SNR_mbh  = np.sqrt(4*np.sum((np.abs(data_mbh_AET[0]).get()*np.abs(h)**2)**2/psd_estimated*df +(np.abs(data_mbh_AET[1]).get()*np.abs(h)**2)**2/psd_estimated*df)) # to check the snr

        print('SNR A estimated mbh:',SNR_mbh )
    
        if (samp.iteration<max_it_update):

                if i*mem % 2 ==0:

                    print("--------- glitch moves updates ----------------")
                
                    gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-10)

                    gmm_glitch.fit(item_samp_glitch_no_nans)

                    #### compute glitch covariance matrix ###
                    
                    for move in samp.moves:
                        if hasattr(move, "name") and move.name == "selected covariance":
                            move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)

                
                    print("---------- noise and mbh moves updates -----------------")
                    gmm_noise = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-25)
                    gmm_noise.fit(noise_sampler)
                    covariances_noise = gmm_noise.covariances_
                    samp.moves[3].all_proposal['noise'].scale = covariances_noise.squeeze()
                    
                    gmm_mbh = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-15)
                    gmm_mbh.fit(item_samp_mbh)
                    covariances_mbh = gmm_mbh.covariances_
                
                    samp.moves[8].all_proposal['mbh'].scale = covariances_mbh.squeeze()


            
        if (samp.iteration==max_it_update):
            print('final update...')
            mem = int(samp.iteration*0.9)
                            
            gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-10)

            gmm_glitch.fit(item_samp_glitch_no_nans)

            #### compute glitch covariance matrix ###

            for move in samp.moves:
                if hasattr(move, "name") and move.name == "selected covariance":
                    move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)

        
            print("---------- noise and mbh moves updates -----------------")
            gmm_noise = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-25)
            gmm_noise.fit(noise_sampler)
            covariances_noise = gmm_noise.covariances_
            samp.moves[3].all_proposal['noise'].scale = covariances_noise.squeeze()
            
            gmm_mbh = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-15)
            gmm_mbh.fit(item_samp_mbh)
            covariances_mbh = gmm_mbh.covariances_
        
            samp.moves[8].all_proposal['mbh'].scale = covariances_mbh.squeeze()


   
        return False
          

fp = "analazing_full_spritz_data_less_walkers_smaller_prior.h5"

if fp in os.listdir():
    print('try to get last sample')
    last_state =  HDFBackend(fp).get_last_sample()
    new_coords = last_state.branches_coords.copy()
    print('backhand')
    # make sure that there are not nans
    for el in branch_names:
        inds[el]  = last_state.branches_inds[el]
        new_coords[el][~inds[el]] = coords[el][~inds[el]]
        coords[el] = new_coords[el].copy()


ensemble = EnsembleSampler(
        nwalkers,
        ndims,
        log_like_fn,
        priors,
        args=[transform_fn, fft_data_cutted ,df,freqs_cut,time,dt,h],
        tempering_kwargs=tempering_kwargs,
        moves=moves,
        rj_moves=rj_moves_mix,
        provide_groups=True,
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        branch_names=branch_names,
        update_iterations=20,
        update_fn=update_fn,
        nbranches=3,
        vectorize=True,
        track_moves =True,
        backend=(fp))


nsteps =20000
thin_by=1
burn=0
    

print('start')


log_prior = ensemble.compute_log_prior(coords, inds=inds)

log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]

## setup branch supplemental to carry group stretch information
closest_inds = -np.ones((ntemps, nwalkers, nleaves_max["glitch"], nfriends), dtype=int)
closest_inds_cov = -np.ones((ntemps, nwalkers, nleaves_max["glitch"]), dtype=int)

closest_inds_mbh = -np.ones((ntemps, nwalkers, nleaves_max["mbh"], nfriends), dtype=int)
closest_inds_cov_mbh = -np.ones((ntemps, nwalkers, nleaves_max["mbh"]), dtype=int)

from eryn.state import BranchSupplemental

branch_supps = {
    "glitch": BranchSupplemental(
        {"inds_closest": closest_inds, "inds_closest_cov": closest_inds_cov}, base_shape=(ntemps, nwalkers, nleaves_max["glitch"])
    ),
    "mbh":BranchSupplemental(
        {"inds_closest": closest_inds_mbh, "inds_closest_cov": closest_inds_cov_mbh}, base_shape=(ntemps, nwalkers, nleaves_max["mbh"])
    ),
}

start_state = State(coords, inds=inds , log_like=log_like, log_prior=log_prior, branch_supplemental=branch_supps)

out = ensemble.run_mcmc(start_state, nsteps, burn=burn, progress=True, thin_by=thin_by)





