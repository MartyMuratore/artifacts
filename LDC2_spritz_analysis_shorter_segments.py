# Glitch search pipeline to analyze the Light Spritz data set

### import python packages packages

import numpy as np
import os
import matplotlib.pyplot as plt
import corner

from chainconsumer import ChainConsumer

### import cupy

import cupy as cp

### import packages for signal processing and covariance matrices computations

import scipy.stats as stats
import scipy.signal
import scipy.signal as signal
from scipy.signal import welch
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import multivariate_normal
from scipy.stats import norm
from scipy import interpolate

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

### import Eryn 

from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer, SearchConvergeStopping, Stopping
from eryn.backends import HDFBackend
from eryn.moves import GaussianMove, StretchMove, GroupStretchMove , GroupMove, ReversibleJumpMove,DistributionGenerateRJ,MTDistGenMoveRJ, MTDistGenMove

### import lisatools

from lisatools.Gaussian_prior import gaussian_dist
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.glitch_shapelet_analytical_waveform import combine_shapelet_link12_frequency_domain, tdi_shapelet_link12_frequency_domain,tdi_shapelet_link12,tdi_glitch_link12
from lisatools.group_stretch_proposal import MeanGaussianGroupMove as group_stretch
from lisatools.group_stretch_proposal import SelectedCovarianceGroupMove
from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity

### import BBhx

from bbhx.utils.constants import *
from bbhx.utils.transform import *
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse


## This command is needed to set which GPU to use

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



plt.style.use('revtex_base.mplstyle')

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(0)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using or not
use_gpu = True

if use_gpu is False:
    xp = np


### setting number and name of branches
branch_names = ["glitch","noise"]

### setting the dimensionality of the parameter to fix
ndims = {"glitch": 3, "noise": 3} 

### setting the Reversible Jump on the glitches allowing between 1 and 0 glitches
nleaves_max = {"glitch": 1, "noise": 1} 
nleaves_min = {"glitch": 0, "noise": 1} 

### setting the number of walkers to use and the number of temperatures 

nwalkers = 25
ntemps = 10
Tmax = np.inf
tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax) # here the maximum temperature is the to infinite so that we ensure sampling the priors ( see https://arxiv.org/abs/2303.02164 )


### defining the log-likelihood the input of the function are:
### 1) a 2D vector with the number of parameters to fit
### 2) the Spritz data
### 3) the sampling frequency
### 4) the frequency to which the likelihood needs to be evaluated
### 5) the value of the filter to be applied to the likelihood to avoid likage


def log_like_fn(x_param, groups, data, df, freqs, filter_tf):   

    glitch_params_all, beta_params_all =  x_param

    group_glitch, group_beta  = groups ## this group function is needed for the Reversible jump, it is important to keep the same order of the vector of the parameter in imput

    #  ------------------------- #
    ## this loop is needed because we only do reversible jump on the glitches and not on the noise parameters, therfore this avoid the likelihood to output a vector of null components in case the
    ## zero glitch hypothesis is favoured
    
    ngroups = int(group_beta.max()) + 1

    group_glitch_dict = {}
   
    for i in range(ngroups):
        index = np.where(group_glitch == i)
        if len(index) > 0:
                group_glitch_dict[i] = glitch_params_all[index]
        else:
            group_glitch_dict[i] = xp.zeros((1, 3))
   
    logl_all = []

    #  ------------------------- #
    
    for group in range(ngroups):  
        
        shapelet_params = group_glitch_dict[group]

        beta_params = beta_params_all[group]

        #-------- power spectral density estimation ---- #

        ## here we estimate the PSD for second generation TDI from the parametrized noise model with three amplitude parameters
        
        psd_estimated = noise_models_spritz(freqs, isi_rfi_back_oms_noise_level =beta_params[0],   tmi_oms_back_level =beta_params[1],acc_level =beta_params[2], T = 8.322688660167833)

        ## we need to apply the filter value to the PSD 
        tot_psd =  xp.asarray([psd_estimated* np.abs(filter_tf)**4,  psd_estimated* np.abs(filter_tf)**4])
        

        ## ------ we estimate here the shapelet tampleate for second generation TDI --------- #

        ### we compute X,Y,Z
        
        templateXYZ = combine_shapelet_link12_frequency_domain(freqs,shapelet_params,T = 8.322688660167833, tdi_channel='second_gen').squeeze()

        ### we compute A,E,T
        
        A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2]) ## to account for the filter
        
        fft_template = xp.asarray([A* np.abs(filter_tf)**2, E* np.abs(filter_tf)**2]) 
        

        ## in case you want to see the templeate of the signal versus the data 
        plots = False
        if plots==True:
            plt.figure()
            plt.loglog(freqs,np.abs(fft_template.get()[0]),'k',label ='templeate')
            plt.loglog(freqs,np.abs(data.get()[0]),'b',label = 'data')
            plt.legend()
            plt.grid()
            plt.savefig("templeate.png")
            plt.close()
    
     
        xp.get_default_memory_pool().free_all_blocks()

        ## ------ Evaluation of the log-likelihood  ---------- ##
        
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


wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=False),
    response_kwargs=dict(TDItag="AET"),   
    use_gpu=use_gpu)

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

##  --------------  time domain data to use for the analysis summing the MBHB and the light-Spritz --------------  ##

A_data,E_data, T_data = AET(data_tdi_X, data_tdi_Y, data_tdi_Z)

A_data_tot = A_data + data_channels_AET_TD.get()[0]
E_data_tot = E_data + data_channels_AET_TD.get()[1]
T_data_tot = T_data + data_channels_AET_TD.get()[2]


# ---------------------- PLOTS --------------------#
plt.figure()
plt.title('TDI A:  one black hole binaries in noisy data')
plt.plot(time,data_channels_AET_TD.get()[0], label="glitch ,  noise and GW  A", alpha=0.9)
plt.ylabel("TDIs")
plt.xlabel("Time [s]")
#plt.xlim([2.625*1e6,2.630*1e6])
plt.grid()
plt.savefig("time_domain_spritz_mbh_data.png", dpi=300)
plt.legend()
plt.figure()
plt.title('TDI A: Three glitches and one black hole binaries in noisy data')
plt.plot(time,A_data_tot, label="glitch ,  noise and GW  A", alpha=0.9)
plt.plot(time,E_data_tot, label="glitch , noise and GW E ", alpha=0.8)
plt.ylabel("TDIs")
plt.xlabel("Time [s]")
#plt.xlim([6*1e5,6.1*1e5])
plt.grid()
plt.savefig("time_domain_spritz_data.png", dpi=300)
plt.legend()

## ----- plot the time frequency  ------ ##
# Choose the length of averaging segments
nperseg = int(6e3 / 5)
# Compute spectrogram
fxx, txx, Sxx = scipy.signal.spectrogram(A_data_tot, 
                                         fs=1.0/dt, 
                                         nperseg=nperseg,
                                         scaling='density',
                                         return_onesided=True)

plt.figure(figsize=(14, 8))
plt.pcolormesh(txx, fxx[fxx > 0], np.log10(Sxx[fxx > 0]), shading='gouraud')

# Set axis labels with larger font sizes
plt.ylabel('Frequency [Hz]', fontsize=20)
plt.xlabel('Time [sec]', fontsize=20)

# Adjust the tick label size for both axes
plt.tick_params(axis='both', which='major', labelsize=16)

# Format the x-axis to ensure consistent scientific notation font size
plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

# Use a logarithmic scale for the y-axis
plt.yscale('log')

# Save and display the figure
plt.savefig("time_frequency_spritz.png")
plt.show()

#### --------------   choise the time (chunks) to consider for the analysis ------------------####

start_time = time[0]  # Start time of the time vector (seconds)
end_time = time[-1]   # End time of the time vector (seconds)
sampling_rate =1/(np.diff(time)[0] )       # Sampling rate (Hz)
block_duration = 90000 # Block duration (seconds) # 100000 day approx  
samples_per_block = block_duration * sampling_rate
total_duration = end_time - start_time
total_blocks = int(total_duration // block_duration)

## -------------- pick the number of segment of interest considering that there are a total of 29 segments and take the appropriate index -----##

nn =21
block_start_time = start_time + nn * block_duration
block_end_time = block_start_time + block_duration

## allowing an overlap among block of 20%

overlap = block_duration * 0.2  
block_start_time -= overlap  # Extend the block start earlier
block_end_time += overlap  # Extend the block end later

# Ensure the start and end times are within the dataset bounds

block_start_time = max(block_start_time, start_time)
block_end_time = min(block_end_time, end_time)

block_start_index = int((block_start_time - start_time) * sampling_rate)
block_end_index = int((block_end_time - start_time) * sampling_rate)

start_index = 0
time = time[block_start_index+start_index :block_end_index]  # First column (time)

## ----------- chunking the time series -----###

A_data= A_data_tot[block_start_index+start_index :block_end_index]  # Second column (TDI channel X)
E_data = E_data_tot[block_start_index+start_index :block_end_index]  # Third column (TDI channel X)
T_data = T_data_tot[block_start_index+start_index :block_end_index]  # Forth column (TDI channel X)

### -------------------- Plots --------------- ##

plt.figure()
plt.title('TDI A: Three glitches and one black hole binaries in noisy data')
plt.plot(time,A_data, label="glitch ,  noise and GW  A", alpha=0.9)
plt.ylabel("TDIs")
ax = plt.gca()
plt.xlabel("Time [s]")
plt.grid()
plt.savefig("time_domain_spritz_data_strech.png", dpi=300)
plt.legend()

## -----------  applied filter to the data to avoid likeage --------   ##

## Define filter parameters
sampling_rate = 1/dt  # Hz
nyquist_freq = sampling_rate / 2
cutoff_freq = 1e-3  #
normalized_cutoff = cutoff_freq / nyquist_freq

## Design a first-order Butterworth filter
b, a = butter(N=1, Wn=normalized_cutoff, btype='low', analog=False)

A_data_filtered = filtfilt(b, a, A_data)[500:-500]
E_data_filtered = filtfilt(b, a, E_data)[500:-500]

time = time[500:-500]

##  ------------- frequency data to be use for the analysis----------------  ##

# Note we only use A and E

freqs = np.fft.rfftfreq(len(A_data_filtered), dt)  # fs =1/dt

Anfft = np.fft.rfft(A_data_filtered ) * dt # TD glitch
Enfft = np.fft.rfft(E_data_filtered) * dt # TD glitch

fft_data_AE = xp.array([Anfft,Enfft])  

## ---------------------- noise models ---------------- ##

def noise_models_spritz(f,  isi_rfi_back_oms_noise_level = np.sqrt( (6.35e-12)**2 + (3.32e-12)**2+ (3.0E-12)**2 ),    tmi_oms_back_level = np.sqrt( (1.42e-12)**2 +(3.0E-12)**2 ),  acc_level = 2.4e-15,    T = 8.322688660167833):

    # Common TDI factor for first gen TDI, which can be factorized as (1 - D^2) * X_0, with X_0 being a simple Michelson.
    Cxx = (np.abs(1 - np.exp(-2j*np.pi*f*T)**4)*np.abs(1 - np.exp(-2j*np.pi*f*T)**2))**2

    # conversion factors into ffd units used in LDC
    lamb = 1064.5e-9
    c = 299792458.0
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


## --------- Reducing the analyzed frequencies ---------- ##

df = freqs[1] - freqs[0]  # 1 / (dt * len(t_in))

fmin = 2e-5
fmax = 2e-2
frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL

freqs_cut =  np.array(freqs[frequencymask])

## --------------------------  Get the filter's frequency response -------------  ##
_, h = freqz(b, a, worN=len(freqs), fs=1/dt)

## ------------- Apply the filter in the frequency domain to the unfiltered noise PSD ------------ ##

Sa_unfiltered = noise_models_spritz(freqs) 
Se_unfiltered =  noise_models_spritz(freqs)  

Sa_filtered = Sa_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response
Se_filtered = Se_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response

h = h[frequencymask]

fft_data_cutted = xp.array([fft_data_AE[0,:][frequencymask],fft_data_AE[1,:][frequencymask] ]) ## these are the final data used for the analysis

## ------- visualizing the data in frequency and time domain ------ ##

freq_welch, psd_data_E = welch(E_data, fs=1/dt, window=('kaiser',15), nperseg=len(E_data)//2, noverlap=50)
freq_welch, psd_data_A = welch(A_data, fs=1/dt, window=('kaiser',15), nperseg=len(E_data)//2, noverlap=50)
freq_welch_f, psd_data_A_filtered= welch(A_data_filtered, fs=1/dt, window='boxcar', nperseg=len(A_data_filtered)//2, noverlap=50)
freq_welch_f, psd_data_E_filtered= welch(E_data_filtered, fs=1/dt, window='boxcar', nperseg=len(E_data_filtered)/2, noverlap=50)


plt.figure()
plt.loglog(freq_welch_f[1:],np.sqrt( psd_data_A_filtered)[1:],'b',label='psd data A')
plt.loglog(freq_welch_f[1:],np.sqrt(psd_data_E_filtered)[1:],'y',label='psd data E ')
plt.loglog(freq_welch[1:],np.sqrt(psd_data_E)[1:],'r',label='psd data welch E')
plt.loglog(freq_welch[1:],np.sqrt(psd_data_A)[1:],'m',label='psd data welch A')
plt.loglog(freqs[1:],np.sqrt(Sa_filtered)[1:],'k--',label='model PSD E filtered')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Fractional frequency [strain]')
plt.legend()
plt.grid()          
plt.savefig("spritz_data_psd_vs_model.png")
plt.close()


if __name__ == "__main__":

    Tobs = time[-1]
    
    #### ------- priors ------- ###

    priors = {}

    ## ------------- glitches  priors ---------##
    
    priors_glitch = {
    0: uniform_dist(time[0],Tobs),  
    1: uniform_dist(-35,-20), 
    2: uniform_dist(1, 1e5)} 

    
    priors["glitch"] = ProbDistContainer(priors_glitch)

    ## -------------noise priors----------##
    
    priors_noise = {
    0: uniform_dist((7e-12),(8e-12)),
    1: uniform_dist((2.5e-12),(3.5e-12)),
    2: uniform_dist((2e-15),(3e-15)),}
   
    priors['noise'] = ProbDistContainer(priors_noise) 

    ## ---------------- moves ------------------ ##

    # Defining the covariance matrix for the gaussian move in the in model move for glitches  
    
    nfriends = nwalkers

    gibbs = []
    for i in range(nleaves_max["glitch"]):
        tmp = np.zeros((nleaves_max["glitch"], ndims["glitch"]), dtype=bool)
        tmp[i] = True

        gibbs.append(("glitch", tmp))
   
    gs = group_stretch(nfriends=nfriends,gibbs_sampling_setup=gibbs,  n_iter_update=100) #
    scgm = SelectedCovarianceGroupMove(nfriends=1, gibbs_sampling_setup=gibbs,n_iter_update=100)

    # noise move
        
    factor = 0.00001
    cov = { 
    "noise": np.diag(np.ones(ndims['noise'])) * factor }

    # total moves glitches and noise
 
    moves = [ (gs, 0.5), (scgm, 0.2),(StretchMove(gibbs_sampling_setup ="noise"),0.3),(GaussianMove(cov,gibbs_sampling_setup ="noise"),0.3)] 
    
    
    ### ---- coordinates definition -------- ###

    coords = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"], ndims["glitch"])),
            "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"])),
            }

    ### ---------  coordinates for the glitches --------- ###

    for nn in range(nleaves_max["glitch"]):

        coords["glitch"][:, :, nn] =priors["glitch"].rvs(size=(ntemps, nwalkers,))
            
    coords["noise"] = priors["noise"].rvs(size=(ntemps, nwalkers,nleaves_max["noise"]))


    
    ## ------------ indices to start the reversible jump we need to shaffle the index around to inizializate the glitch-walkers (e.g. glitch present in the data)  ----###
    inds = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"]), dtype=bool),
            "noise": np.ones((ntemps, nwalkers, nleaves_max["noise"]),  dtype=bool)
        }
  

    for nwalk in range(nwalkers):
        for temp in range(ntemps): 
            nl = np.random.randint(nleaves_min["glitch"], nleaves_max["glitch"]+1) 
            inds_tw=np.zeros(nleaves_max["glitch"],dtype=bool) 
            inds_tw[:nl]=True
            np.random.shuffle(inds_tw)
            inds['glitch'][temp,nwalk]=inds_tw
   
 
def update_fn(i, res, samp):
        max_it_update=50000
        mem =800


        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()[-mem:,0]))
        print("min last loglike",np.min(samp.get_log_like()[-mem:,0]))
    
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
            print("rj \n",samp.rj_acceptance_fraction[0] )
            print("swap \n",samp.swap_acceptance_fraction)


        ####  ----------- noise posterios plots --------- #######
    
        noise_sampler= samp.get_chain()["noise"][-mem:,0].reshape(-1,ndims['noise'])
        c = ChainConsumer()
        parameter_labels = ['$isi+rfi_OMS$','$tmi$','$TM$']
        c.add_chain(noise_sampler, parameters=parameter_labels, name='noise', color='#6495ed')
        c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
        c.add_marker([ 7.768197989237916e-12, 3.3190962625389463e-12 ,  2.4e-15], marker_style="x", marker_size=500, color='#DC143C')
        fig = c.plotter.plot(figsize=(8,8), legend=True)
        plt.savefig("noise_spritz.png", dpi=300)
        plt.close()

        ####  ----------- likelihood plots --------- #######

    
        likelihood = samp.get_log_like()[-mem:, 0,:]
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.xlabel('iter')
        plt.ylabel('Likelihood (single glitch)')
        plt.title('likelihood evolutions for all the walkers at 0 temperature with glitch')
        plt.grid(True)
        plt.show()
        plt.savefig("likelihood_spritz_data_glitch_search.png", dpi=300)
        plt.close()


        ####  ----------- glitch plots --------- #######

        
        item_samp_glitch_no_nans= samp.get_chain()["glitch"][-mem:,0][~np.isnan(samp.get_chain()["glitch"][-mem:,0])].reshape(-1,3)
        nleaves = samp.get_nleaves()['glitch'][:,0,:].reshape(-1)
        nleaves_with_1 = np.sum(nleaves == 1)
        fraction1leaves = nleaves_with_1/len(nleaves)

        # check if there are glitches in the data if ther are compute the glitch SNR 

        if fraction1leaves>0.01:

            snr_posterior = []
            n_iter = 3000
            n_samples_glitch = item_samp_glitch_no_nans.shape[0]
            n_samples_noise = noise_sampler.shape[0]
            min_samples=np.min([n_samples_glitch,n_samples_noise])
            
            for _ in range(n_iter):
                # Pick a single common random index
                idx = np.random.choice(min_samples, size=1, replace=False)[0]
                
                # Extract noise parameters and shapelet parameters from the same index
                noise_params = noise_sampler[idx,:]  # shape (3,)
                shapelet_params = item_samp_glitch_no_nans[idx,:]  # shape (3,)

                # Compute estimated PSD
                psd_estimated = (
                    noise_models_spritz(
                        freqs_cut,
                        isi_rfi_back_oms_noise_level=noise_params[0],
                        tmi_oms_back_level=noise_params[1],
                        acc_level=noise_params[2]
                    ) * np.abs(h)**4
                )

                # Generate glitch template
                templateXYZ = tdi_shapelet_link12_frequency_domain(
                    freqs_cut,
                    tau=shapelet_params[0],
                    Deltav=shapelet_params[1],
                    beta=shapelet_params[2],
                    tdi_channel='second_gen'
                ).squeeze()

                A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2])

                # Compute SNR
                SNR_estimated = np.sqrt(
                    4 * np.sum(
                        (np.abs(A)**2 * np.abs(h)**4 + np.abs(E)**4 * np.abs(h)**2) / psd_estimated * df
                    )
                )

                snr_posterior.append(SNR_estimated)

            # Final results
            print("SNR values (posterior):", snr_posterior)
            c = ChainConsumer()
            parameter_labels = ['$SNR-values$']
            snr_posterior = np.array(snr_posterior)
            c.add_chain(snr_posterior, parameters=parameter_labels , name='SNR values', color='#6495ed')
            c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=False, serif=True)
            fig = c.plotter.plot(figsize=(8,8), legend=True)
            plt.savefig("snr_posteriors.png", dpi=300)
            plt.close()

            print("Mean estimated SNR:", np.mean(snr_posterior))


            plot_psd = True
            if plot_psd == True:
                plt.figure(figsize=set_figsize('single', ratio =0.8))
                #plt.loglog(freq_welch,np.sqrt(psd_data_A),'b-',label='psd data A welche')
                plt.loglog(freq_welch_f,np.sqrt(psd_data_A_filtered),'b-',label='Filtered-data PSD TDI $A$')
                plt.loglog(freq_welch_f,np.sqrt(psd_data_E_filtered),'r-',label='Filtered-data PSD TDI $E$')
                #plt.loglog(freq_welch_f,np.sqrt(psd_data_E_filtered),'g-',label='psd data E filtered')
                plt.loglog(freqs_cut,np.sqrt(psd_estimated),'k-',label='Estimated PSD TDI $A/E$')
                plt.grid(True)
                plt.xlabel('Frequency [Hz]')
                plt.ylabel('TDI [$1/\sqrt{Hz}$]')
                plt.ylim([8e-24,4e-21])
                plt.xlim([5e-5,2e-2])
                # Enable grid with subtle dashed lines
                # Place the legend outside the plot for better clarity
                plt.legend(loc="best")
                plt.legend()    
                plt.savefig("estimated_model.png")
                plt.close()
          
        if (samp.iteration<max_it_update):
    
            if fraction1leaves>0.6:
                ('printing glitch updates')
                c = ChainConsumer()
                parameter_labels = ['$t0$', '$ln(DV)$', '$\\tau$']
                c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
                
                c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
                fig = c.plotter.plot(figsize=(8,8), legend=True)
                plt.savefig("shapelet_corner_plot_spritz_glitch_search.png", dpi=300)
                plt.close()
   
                gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-20)

                gmm_glitch.fit(item_samp_glitch_no_nans)

                #### compute glitch covariance matrix ###
            
                for move in samp.moves:
                    if hasattr(move, "name") and move.name == "selected covariance":
                        move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)
                
            gmm_noise = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-25)
            gmm_noise.fit(noise_sampler)
            covariances_noise = gmm_noise.covariances_
        
            for mm, el_cov in zip(samp.moves[3:], covariances_noise):
                mm.all_proposal['noise'].scale = el_cov
            
        if (samp.iteration==max_it_update):
            mem = int(samp.iteration*0.9)

            if fraction1leaves>0.6:
                                    
                bay_gmm = BayesianGaussianMixture(n_components=nleaves_max['glitch'], n_init=10)

                bay_gmm.fit(item_samp_glitch_no_nans[-mem:, 0][:,None])  
                
                bay_gmm_weights = bay_gmm.weights_
                n_clusters_ = (np.round(bay_gmm_weights, 2) > 0).sum()
                
                print('Estimated number of clusters: ' + str(n_clusters_))

                ### now in based at the estimated cluster i compute the covariance ##

                gmm_glitch = GaussianMixture(n_components=n_clusters_,covariance_type='full', tol=0.00001, reg_covar=1e-20)

                gmm_glitch.fit(item_samp_glitch_no_nans)

                c = ChainConsumer()
                parameter_labels = ['$t0$', '$ln(DV)$', '$\\tau$']
                c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
                
                c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
                fig = c.plotter.plot(figsize=(8,8), legend=True)
                plt.savefig("shapelet_corner_plot_spritz_glitch_search.png", dpi=300)
                plt.close()

                for move in samp.moves:

                    if hasattr(move, "name") and move.name == "selected covariance":
                        move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)

            gmm_noise = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-50)
            gmm_noise.fit(noise_sampler)
            covariances_noise = gmm_noise.covariances_
          

            for mm, el_cov in zip(samp.moves[3:], covariances_noise):
                mm.all_proposal['noise'].scale = el_cov

      
        return False

            
    
def stop_fn(i, res, samp):

    nleaves = samp.get_nleaves()['glitch'][:,0,:].reshape(-1)

    #Count the number of 0s
    nleaves_with_0 = np.sum(nleaves == 0)

    #Count the number of 1s
    nleaves_with_1 = np.sum(nleaves == 1)


    fraction0leaves = nleaves_with_0/len(nleaves)
    print(f"Fraction of leaves with 0: {fraction0leaves}")

    fraction1leaves = nleaves_with_1/len(nleaves)
    print(f"Fraction of leaves with 1: {fraction1leaves}")

    if fraction0leaves >= 0.7:
       print('there are no glitches in the data')
       return True
    else:
        return False            

#fp =  "analazing_spritz_data_glitch28.h5" # "analazing_spritz_data_glitch29_only_mbh.h5" # "analazing_spritz_data_glitch23.h5" # "analazing_spritz_data_glitch6.h5"

fp = 're-analazing_spritz_data_glitch21.h5' #"re-analazing_spritz_data_glitch7_less_strongerfilt.h5" "re-analazing_spritz_data_glitch29SNR.h5"


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
        args=[fft_data_cutted ,  df,freqs_cut,h] ,#[wave_gen, bbh_kwargs, transform_fn, fft_data_cutted , psd, df,freqs_cut,time, dt],
        tempering_kwargs=tempering_kwargs,
        moves=moves ,
        rj_moves=True,
        provide_groups=True,
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        branch_names=branch_names,
        update_iterations=1,
        update_fn=update_fn,
        #stopping_fn=stop_fn,
        stopping_iterations=1,
        nbranches=2,
        vectorize=True,
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

branch_supps = {
    "glitch": BranchSupplemental(
        {"inds_closest": closest_inds, "inds_closest_cov": closest_inds_cov}, base_shape=(ntemps, nwalkers, nleaves_max["glitch"])
    ),
    "mbh": None
}

start_state = State(coords, inds=inds , log_like=log_like, log_prior=log_prior, branch_supplemental=branch_supps)

out = ensemble.run_mcmc(start_state, nsteps, burn=burn, progress=True, thin_by=thin_by)


