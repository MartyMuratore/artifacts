### import python packages packages

import numpy as np
import os
import matplotlib.pyplot as plt
import corner

from chainconsumer import ChainConsumer


### import packages for signal processing and covariance matrices computations

from scipy.signal import welch
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import norm
from scipy.interpolate import CubicSpline

from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

### import Eryn 

from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import TransformContainer, SearchConvergeStopping, Stopping
from eryn.backends import HDFBackend
from eryn.utils import SearchConvergeStopping
from eryn.moves import GaussianMove, StretchMove, GroupStretchMove , GroupMove, ReversibleJumpMove,DistributionGenerateRJ,MTDistGenMoveRJ, MTDistGenMove

### import lisatools

from lisatools.sampling.likelihood import Likelihood
import sys
sys.path.append("../")
#from utility_funcs.sampling_funcs.group_stretch_proposal import MeanGaussianGroupMove as group_stretch

from lisatools.group_stretch_proposal import MeanGaussianGroupMove as group_stretch

from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(6)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

DELETE_BACKEND = False
# whether you are using or not
use_gpu = False

if use_gpu is False:
    xp = np

### setting number and name of branches
branch_names = ["noise"]

### setting the dimensionality of the parameter to fix
ndims = {"noise": 6} 

### setting the Reversible Jump on the glitches allowing between 1 and 0 glitches
nleaves_max = {"noise": 1} 
nleaves_min = { "noise": 1} 

### setting the number of walkers to use and the number of temperatures 

nwalkers = 12
ntemps = 1
# Tmax = np.inf
Tmax = 1.0
tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax) # here the maximum temperature is the to infinite so that we ensure sampling the priors ( see https://arxiv.org/abs/2303.02164 )
                        
   
parameter_noise_amplitude = [2.4e-15, 1.42e-12,3.32e-12,6.35e-12,3.0e-12,3.0e-12]
armlength = 8.322688660167833
true_start = True
## -------------------   uploading Spritz data set ------------------ ##
import h5py

# Open the HDF5 file in read mode
with h5py.File('LDC2_Spritz_only_noise.h5', 'r') as f:
    # Access the dataset named 'time_series'
    data = f['time_series'][:]
    
    # Split the data into time and TDI channel X
    time = data[500:, 0] - data[500:, 0][0] # First column (time) #NB I am shifting the time to start at 0
    data_tdi_X = -data[500:, 1]  # Second column (TDI channel X)
    data_tdi_Y = -data[500:, 2]  # Third column (TDI channel Y)
    data_tdi_Z = -data[500:, 3]  # Forth column (TDI channel Z)


dt = time[1]-time[0]

####### ---------- defining the GAP mask ---------- ###

# maskgapX = np.where((data_tdi_X==0),0,1).astype(np.int32)
# maskgapY = np.where((data_tdi_Y==0),0,1).astype(np.int32)
# maskgapZ = np.where((data_tdi_Z==0),0,1).astype(np.int32)


freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt
## -----------  applied filter to the data to avoid likeage --------   ##

## Define filter parameters
sampling_rate = 1/dt  # Hz
nyquist_freq = sampling_rate / 2
cutoff_freq = 1e-3  #
normalized_cutoff = cutoff_freq / nyquist_freq

## Design a first-order Butterworth filter
b, a = butter(N=1, Wn=normalized_cutoff, btype='low', analog=False)


X_data_filtered = filtfilt(b, a, data_tdi_X)[500:-500]
Y_data_filtered = filtfilt(b, a, data_tdi_Y)[500:-500]
Z_data_filtered = filtfilt(b, a, data_tdi_Z)[500:-500]

time = time[500:-500]


##  ------------- frequency data to be use for the analysis----------------  ##

# Note we only use A and E

freqs = np.fft.rfftfreq(len(X_data_filtered), dt)  # fs =1/dt

Xnfft = np.fft.rfft(X_data_filtered ) * dt # TD glitch
Ynfft = np.fft.rfft(Y_data_filtered) * dt # TD glitch
Znfft = np.fft.rfft(Z_data_filtered) * dt # TD glitch

fft_data_XYZ = xp.array([Xnfft,Ynfft,Znfft])  

## ---------------------- noise models ---------------- ##

# conversion factors into ffd units used in LDC
lamb = 1064.5e-9
c = 299792458.0

def noise_models_correlation_spritz(f,  
                        acc_level = 2.4e-15,
                        readout_tmi = 1.42e-12,
                        readout_rfi = 3.32e-12,
                        readout_isi = 6.35e-12,
                        backlink_tmi = 3.0e-12,
                        backlink_rfi = 3.0e-12,
                        T = armlength):

    # Common TDI factor for first gen TDI, which can be factorized as (1 - D^2) * X_0, with X_0 being a simple Michelson.
    omega = 2 * np.pi * f
    Cxx = 16 * np.sin(omega*T)**2 * np.sin(2*omega*T)**2
    Cxy = -16 * np.sin(omega*T) * np.sin(2*omega*T)**3

    nu0 = c / lamb
    
    # conversion: divide by lambda to get cycles, take a derivative to get frequency in Hz, divide by nu0 to get ffd
    disp_2_ffd = (omega / lamb / nu0)**2
    # conversion: divide by lambda to get cycles/s^2, integrate to get frequency in Hz, divide by nu0 to get ffd
    acc_2_ffd = (1 / (lamb * omega ) / nu0)**2


    # TM noises. 
    tm_noise = acc_level**2 * (1 + (0.4e-3/f)**2)

    # Backlink noises
    backlink_tmi_noise = backlink_tmi**2 *(1+ (2e-3 / f)**4)
    readout_tmi_noise = backlink_rfi**2 *(1+ (2e-3 / f)**4)


    # --------- these are the PSD for TM
    TM_transfer_XX = 4 * Cxx * ( 3 + np.cos(2*omega*T))
    # these are the noise terms
    TM_noise_XX_ffd = TM_transfer_XX * acc_2_ffd* tm_noise 

    # --------- these are the CSD for TM
    TM_transfer_XY = 4 * Cxy
    # these are the noise terms
    TM_noise_XY_ffd = TM_transfer_XY * acc_2_ffd* tm_noise

    # --------- these are PSD for OMS 

    TMI_readout_transfer_XX =  Cxx * ( 3 + np.cos(2*omega*T))
    ISI_readout_transfer_XX = 4* Cxx
    RFI_readout_transfer_XX = 4* Cxx 
    # these are the noises in TDI
    TMI_readout_transfer_XX_ffd = TMI_readout_transfer_XX * disp_2_ffd * readout_tmi_noise**2
    ISI_readout_transfer_XX_ffd = ISI_readout_transfer_XX * disp_2_ffd * readout_isi**2
    RFI_readout_transfer_XX_ffd = RFI_readout_transfer_XX * disp_2_ffd * readout_rfi**2

    # --------- these are CSD for OMS

    TMI_readout_transfer_XY =  Cxy
    ISI_readout_transfer_XY = Cxy
    RFI_readout_transfer_XY = Cxy 
    # these are the noises in TDI
    TMI_readout_transfer_XY_ffd = TMI_readout_transfer_XY * disp_2_ffd * readout_tmi_noise**2
    ISI_readout_transfer_XY_ffd = ISI_readout_transfer_XY * disp_2_ffd * readout_isi**2
    RFI_readout_transfer_XY_ffd = RFI_readout_transfer_XY * disp_2_ffd * readout_rfi**2

    # --------- these are PSD for backlink 

    TMI_backlink_transfer_XX =  TMI_readout_transfer_XX
    RFI_backlink_transfer_XX =  ISI_readout_transfer_XX

    TMI_backlink_transfer_XX_ffd = TMI_backlink_transfer_XX * disp_2_ffd * backlink_tmi_noise**2
    RFI_backlink_transfer_XX_ffd = RFI_backlink_transfer_XX * disp_2_ffd * backlink_rfi**2

    # --------- these are CSD for backlink 

    TMI_backlink_transfer_XY =  TMI_readout_transfer_XY
    RFI_backlink_transfer_XY =  TMI_readout_transfer_XY

    TMI_backlink_transfer_XY_ffd = TMI_backlink_transfer_XY * disp_2_ffd * backlink_tmi_noise**2
    RFI_backlink_transfer_XY_ffd = RFI_backlink_transfer_XY * disp_2_ffd * backlink_rfi**2

    total_noise_XX = RFI_backlink_transfer_XX_ffd + TMI_backlink_transfer_XX_ffd + RFI_readout_transfer_XX_ffd + ISI_readout_transfer_XX_ffd +TMI_readout_transfer_XX_ffd + TM_noise_XX_ffd
    total_noise_XY = RFI_backlink_transfer_XY_ffd + TMI_backlink_transfer_XY_ffd + RFI_readout_transfer_XY_ffd + ISI_readout_transfer_XY_ffd +TMI_readout_transfer_XY_ffd + TM_noise_XY_ffd
    
    #TODO: Be careful with this. 
    total_noise_XX[0] = total_noise_XX[1]
 
    return total_noise_XX,total_noise_XY

## --------- Reducing the analyzed frequencies ---------- ##

df = freqs[1] - freqs[0]  # 1 / (dt * len(t_in))
'''
fmin = 2e-5
fmax = 2e-2
frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL

freqs_cut =  np.array(freqs[frequencymask])

freq_welch_f, psd_data_XX= welch(data_tdi_X, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//120, noverlap=50)

from scipy.signal import coherence
from scipy.signal import csd
f_coh, Cxy_data = csd(data_tdi_X, data_tdi_Y, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//12, noverlap=50)

plt.figure()
plt.loglog(freq_welch_f[1:],np.sqrt(noise_models_correlation_spritz(freq_welch_f)[0])[1:],'-',alpha=1,label = 'PSD data')
plt.loglog(freq_welch_f[1:],np.sqrt(psd_data_XX)[1:],'-',alpha=1,label = 'PSD model')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
#plt.ylim([1e-22,2e-19])
plt.xlim([2.8e-5,freqs[-1]])
plt.savefig("/data/mmuratore/gaps_project/artifacts/code/noise_analysis/martina_noise_analysis/plots/data_vs_model.png")
plt.close()

plt.figure()
plt.loglog(f_coh[1:],np.abs(noise_models_correlation_spritz(f_coh)[1])[1:],'-',alpha=1,label = 'CSD model')
plt.loglog(f_coh[1:],np.abs(Cxy_data)[1:],'-',alpha=1,label = 'CSD data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
#plt.ylim([1e-22,2e-19])
plt.xlim([2.8e-5,freqs[-1]])
plt.legend()
plt.savefig("/data/mmuratore/gaps_project/artifacts/code/noise_analysis/martina_noise_analysis/plots/data_vs_model_csd.png")
plt.close()

## --------------------------  Get the filter's frequency response -------------  ##

freq_welch_f, psd_data_XX_filtered= welch(X_data_filtered, fs=1/dt, window='boxcar', nperseg=len(X_data_filtered)//120, noverlap=50)
f_coh, Cxy_data_filtered = csd(X_data_filtered,Y_data_filtered, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//12, noverlap=50)


## ------------- Apply the filter in the frequency domain to the unfiltered noise PSD ------------ ##

Sxx_unfiltered = noise_models_correlation_spritz(freqs)[0]
Sxy_unfiltered = noise_models_correlation_spritz(freqs)[1] 

Sxx_filtered = Sxx_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response
Sxy_filtered = Sxy_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response
'''

_, h = freqz(b, a, worN=len(freqs), fs=1/dt)
h = h[frequencymask]

'''
## ----------- Plots model vs data -------##

plt.figure()
plt.loglog(freqs[1:],np.sqrt(Sxx_filtered)[1:],'-',alpha=1,label = 'PSD data')
plt.loglog(freq_welch_f[1:],psd_data_XX_filtered[1:],'-',alpha=1,label = 'PSD model filtered')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
#plt.ylim([1e-22,2e-19])
plt.xlim([2.8e-5,freqs[-1]])
plt.legend()
plt.savefig("/data/mmuratore/gaps_project/artifacts/code/noise_analysis/martina_noise_analysis/plots/data_vs_model.png")

plt.figure()
plt.loglog(f_coh[1:],np.abs(Cxy_data_filtered)[1:],'-',alpha=1,label = 'PSD model filtered')
plt.loglog(freqs[1:],np.abs(Sxy_filtered)[1:],'-',alpha=1,label = 'PSD data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
#plt.ylim([1e-22,2e-19])
plt.xlim([2.8e-5,freqs[-1]])
plt.legend()
plt.savefig("/data/mmuratore/gaps_project/artifacts/code/noise_analysis/martina_noise_analysis/plots/data_vs_model_cxy.png")
plt.close()
'''
#fft_data_cutted = xp.array([fft_data_XYZ[0,:][frequencymask],fft_data_XYZ[1,:][frequencymask],fft_data_XYZ[2,:][frequencymask] ]) ## these are the final data used for the analysis
fft_data = xp.array([fft_data_XYZ[0,:],fft_data_XYZ[1,:],fft_data_XYZ[2,:]])
### --------------- Splines definition ---------------- ####

# we consider 5 knots to allow for a sufficient slowly varing PDS

def spline_psd_mod(logf_knots,spline_weights,f_array):
    noise_uncert=CubicSpline(logf_knots, spline_weights,bc_type='natural')
    modnoisevar=np.exp(np.log(10.0)*noise_uncert(np.log10(f_array))) # this is equivalent to 10^x
    
    return(modnoisevar)


### ------ PLOTS with splines of variation ------ ###
logf_knots = np.linspace(np.log10(freqs_cut[0]), np.log10(freqs_cut[-1]), 5)

# Define your Gaussian parameters
mu = 0       # mean
sigma = 0.1  # standard deviation

# Sample from the normal distribution

noise_uncert_weights=norm.rvs(loc=mu, scale=sigma, size=len(logf_knots)) 

noise_uncert=spline_psd_mod(logf_knots,noise_uncert_weights,freqs_cut)

plt.figure()
plt.loglog(freqs_cut,noise_uncert,'-',alpha=1,label='Cubic')
plt.axvline(x=10**np.log10(freqs_cut[0]),c='k',ls='--')
plt.axvline(x=10**-3.5,c='k',ls='--')
plt.axvline(x=10**-3,c='k',ls='--')
plt.axvline(x=10**-2.5,c='k',ls='--')
plt.axvline(x=10**np.log10(freqs_cut[-1]),c='k',ls='--')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
plt.legend()
plt.savefig("plots/estimated_model.png")
plt.close()

### --------------- Splined PSD  ---------------- ####

def splined_psd(f_array, noise_params, logf_knots, spline_weights, average_armlength, spritz=True):
    """
    Construct splined PSD for A/E channels.
    
    Parameters
    ----------
    f_array : array
        Frequency array.
    noise_params : array-like, length 3
        [isi_rfi_back_oms_noise_level, tmi_oms_back_level, acc_level].
    logf_knots : array
        Knot positions in log-frequency space for spline modulation.
    spline_weights : array
        Weights for spline modulation.
    average_armlength : float
        Average armlength T.
    spritz : bool, optional
        Use Spritz noise model (default=True).
    """

    A, B, C,D,F,G = noise_params

    if spritz:
        psd = noise_models_correlation_spritz(
            f_array,
            acc_level =A,
            readout_tmi =B,
            readout_rfi = C,
            readout_isi = D,
            backlink_tmi = F,
            backlink_rfi = G,
            T=average_armlength,
        ) * spline_psd_mod(xp.array(logf_knots), xp.array(spline_weights),xp.array(f_array))
    
    else:
        raise NotImplementedError("Only Spritz model is implemented so far.")

    return psd
'''
psdAE = splined_psd(freqs_cut,parameter_noise_amplitude,logf_knots,noise_uncert_weights, armlength)

## -----------Plots -------##
plt.figure()
plt.loglog(freqs_cut,np.sqrt(psdAE),'r-',alpha=1,label = 'PSD splined')
plt.loglog(freqs_cut,np.sqrt(noise_models_spritz(freqs_cut)),'b--',alpha=1,label ='PSD')
plt.axvline(x=10**logf_knots[0],c='k',ls='--')
plt.axvline(x=10**logf_knots[1],c='k',ls='--')
plt.axvline(x=10**logf_knots[2],c='k',ls='--')
plt.axvline(x=10**logf_knots[3],c='k',ls='--')
plt.axvline(x=10**logf_knots[4],c='k',ls='--')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Splined AE example ') 
plt.legend() 
plt.savefig("plots/estimated_noise_model.png")
plt.close()
### ------- likelihood definition ------- ###
'''

## -----------noise moves-----------##

# TODO: Think of the proposal here
moves = [(StretchMove(gibbs_sampling_setup ="noise",live_dangerously=False))]

###### ---- cords from true value ----###


starting_coords = np.zeros((ntemps * nwalkers, ndims['noise'] ))
coords = { "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"]))}

#true_vals_spline = np.array((ndims['noise'])*[0])


d1 = 0.0001
d2 = 1e-4

starting_coords[:,:] = (parameter_noise_amplitude + d2 * np.random.randn(nwalkers * ntemps,ndims['noise']) )

coords["noise"] = starting_coords.reshape(ntemps, nwalkers, nleaves_max["noise"], ndims['noise'])


## -------------noise priors----------##


priors = {}

a1_min, a1_max = true_params[0]-1, true_params[0]+1
a2_min, a2_max = true_params[1]-1, true_params[1]+1
a3_min, a3_max = true_params[2]-1, true_params[2]+1
a4_min, a4_max = true_params[3]-1, true_params[3]+1
a5_min, a5_max = true_params[4]-1, true_params[4]+1
a6_min, a6_max = true_params[5]-1, true_params[5]+1
prior_amplitude =1


priors_noise = {
0: uniform_dist(a1_min, a1_max),
1: uniform_dist(a2_min, a2_max),
2: uniform_dist(a3_min, a3_max),
3: uniform_dist(a4_min, a4_max),
4: uniform_dist(a5_min, a5_max),
5: uniform_dist(a6_min, a6_max),

priors['noise'] = ProbDistContainer(priors_noise) 

### ---- coordinates definition for the noise from priors-------- ###


#coords = { "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"]))}

#coords["noise"] = priors["noise"].rvs(size=(ntemps, nwalkers,nleaves_max["noise"]))



## ------------ indices to start the reversible jump  ----###
inds = {"noise": np.ones((ntemps, nwalkers, nleaves_max["noise"]),  dtype=bool)}

fp = 'script_to_estimate_psd_csd_with_spritz.h5'

if DELETE_BACKEND:
    if os.path.exists(fp):
        os.remove(fp)
    print(f"Deleted old backend file: {fp}")
else:
    if fp in os.listdir():
        print('try to get last sample')
        last_state = HDFBackend(fp).get_last_sample()
        new_coords = last_state.branches_coords.copy()
        print('backend')

        # make sure there are no NaNs in the selected glitches
        inds = {}
        for el in branch_names:
            inds[el] = last_state.branches_inds[el]
            new_coords[el][~inds[el]] = coords[el][~inds[el]]
            coords[el] = new_coords[el].copy()
    else:
        print("No backend file found. Starting fresh.")
   

def update_fn(i, res, samp):
        max_it_update=20000
        skipp =1e4

   
        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()[-skipp:,0]))
        print("min last loglike",np.min(samp.get_log_like()[-skipp:,0]))
     
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
            #print("rj \n",samp.rj_acceptance_fraction[0] )
            print("swap \n",samp.swap_acceptance_fraction)


        ####  ----------- noise posterios plots --------- #######

        likelihood = samp.get_log_like()[-skipp:, 0,:]
   
        plt.figure(figsize=(10, 6)),
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.xlabel('iter')
        plt.ylabel('Likelihood noise')
        plt.grid(True)
        plt.savefig("plots/noise_likelihood.png", dpi=300)
        plt.close()
        noise_sampler= samp.get_chain()["noise"][-skipp:,0].reshape(-1,ndims['noise'])
        
        parameter_labels = [f'noise_amplitude_{j}' for j in range(0,7)]

        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        colors = ['#6495ed', '#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181', 
                '#aa96da', '#fcbad3', '#a8d8ea', '#ffcb91', '#d4a5a5']

        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()

        
        for i in range(9):
            axes_flat[i].plot(noise_sampler[:, i], 
                            color=colors[i], alpha=0.7, linewidth=0.5)
            axes_flat[i].set_xlabel('Iteration', fontsize=11)
            axes_flat[i].set_ylabel(parameter_labels[i], fontsize=11)
            axes_flat[i].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig("plots/noise_trace.png", dpi=300)
        plt.close()


        c = ChainConsumer()
        # parameter_labels = ['$isi+rfi_OMS$','$tmi$','$TM$']
        c.add_chain(noise_sampler, parameters=parameter_labels, name='noise', color='#6495ed')
        c.configure(
        summary=False,
        bar_shade=True,         # lascia i marginali come linee (non pieni)
        shade=True,              # riempie i plot 2D
        shade_alpha=0.8,         # trasparenza del riempimento
        serif=True,
        usetex=True,
        legend_artists=False,
        label_font_size=17,
        tick_font_size=15, 
        linewidths=2,      # Optional: thinner contour lines
        max_ticks=5,
        bins=10,
        smooth=2,# Optional: fewer ticks for clarity
        )
        # c.add_marker([ 7.768197989237916e-12, 3.3190962625389463e-12 ,  2.4e-15], marker_style="x", marker_size=500, color='#DC143C')
        fig = c.plotter.plot(figsize=(8,8), truth = parameter_noise_amplitude, legend=True)
        plt.savefig("plots/noise_spritz.png", dpi=300)
        plt.close()



        n_iter = 100
        psdA_samples = []
        psdE_samples = []

        n_samples_noise = noise_sampler.shape[0]
        indices = np.random.choice(n_samples_noise, size=n_iter, replace=False)
        '''
        for idx in indices:
            noise_params = noise_sampler[idx,:]  # shape (>=9,)
        
            psdA = splined_psd(freqs_cut,
                            [parameter_noise_amplitude[0], parameter_noise_amplitude[1], parameter_noise_amplitude[2]],
                            logf_knots, noise_params[:5],armlength) * np.abs(h)**4
            psdE = splined_psd(freqs_cut,
                            [parameter_noise_amplitude[0], parameter_noise_amplitude[1], parameter_noise_amplitude[2]],
                            logf_knots, noise_params[5:],armlength) * np.abs(h)**4
        
            psdA_samples.append(psdA)
            psdE_samples.append(psdE)

        # Convert to arrays
        psdA_samples = np.array(psdA_samples)
        psdE_samples = np.array(psdE_samples)

        # Compute median and 90% credible interval
        psdA_median = np.median(psdA_samples, axis=0)
        psdA_low    = np.percentile(psdA_samples, 5, axis=0)
        psdA_high   = np.percentile(psdA_samples, 95, axis=0)

        psdE_median = np.median(psdE_samples, axis=0)
        psdE_low    = np.percentile(psdE_samples, 5, axis=0)
        psdE_high   = np.percentile(psdE_samples, 95, axis=0)

        # --- Plot ---
        plt.figure()

        # A channel
        plt.fill_between(freqs_cut, np.sqrt(psdA_low), np.sqrt(psdA_high),
                        color='r', alpha=0.3, label='PSDA 90% CI')
        plt.loglog(freqs_cut, np.sqrt(psdA_median), 'r-', label='PSDA median')

        # E channel
        plt.fill_between(freqs_cut, np.sqrt(psdE_low), np.sqrt(psdE_high),
                        color='g', alpha=0.3, label='PSDE 90% CI')
        plt.loglog(freqs_cut, np.sqrt(psdE_median), 'g-', label='PSDE median')

        # Reference model
        plt.loglog(freqs_cut, np.sqrt(noise_models_spritz(freqs_cut)* np.abs(h)**4),
                'b--', label='PSD reference')
        plt.axvline(x=10**logf_knots[0],c='k',ls='--')
        plt.axvline(x=10**logf_knots[1],c='k',ls='--')
        plt.axvline(x=10**logf_knots[2],c='k',ls='--')
        plt.axvline(x=10**logf_knots[3],c='k',ls='--')
        plt.axvline(x=10**logf_knots[4],c='k',ls='--')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Estimated ASD [Hz$^{-1/2}$]')
        plt.legend()
        plt.savefig("plots/eryn_estimated_noise_model.png")
        plt.close()
        '''

def log_like_fn(x,data,df,freqs,filter_tf,subset = 1):

        beta_params = x

        inds = np.arange(0, beta_params.shape[0] + 1, subset)
        
        if inds[-1] < beta_params.shape[0]:
            inds = np.concatenate([inds, np.array([beta_params.shape[0]])])
        logl_all = []

        for i in range(len(inds) - 1):

            start = int(inds[i])
            end = int(inds[i + 1])
            A, B, C,D,F,G =beta_params[start:end].squeeze()

            psd_estimate = noise_models_correlation_spritz(
                f_array,
                acc_level =A,
                readout_tmi =B,
                readout_rfi = C,
                readout_isi = D,
                backlink_tmi = F,
                backlink_rfi = G,
                T=average_armlength,
            ) 

    
            cov_matrix = xp.asarray([
                    [psd_estimate[0] * np.abs(filter_tf)**4, psd_estimate[1] * np.abs(filter_tf)**4, psd_estimate[1] * np.abs(filter_tf)**4],
                    [psd_estimate[1] * np.abs(filter_tf)**4, psd_estimate[0] * np.abs(filter_tf)**4, psd_estimate[1] * np.abs(filter_tf)**4],
                    [psd_estimate[1] * np.abs(filter_tf)**4, psd_estimate[1] * np.abs(filter_tf)**4, psd_estimate[0] * np.abs(filter_tf)**4]])
 ## to account for the filter

        
            # xp.get_default_memory_pool().free_all_blocks()
            

            ## computing the likelihood 

            logl = -1/2 * (4*df* xp.sum((xp.conj(data) *(data)).real /(cov_matrix ), axis=0).sum() )
            logl += -  xp.sum(xp.log(cov_matrix), axis=0).sum()
            
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :])
                  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)
        if not isinstance(logl_out, np.ndarray):
            logl_out = xp.asnumpy(logl_out)
        return logl_out

stop = SearchConvergeStopping(n_iters=5, diff=0.01, verbose=True)

from multiprocessing import (get_context,cpu_count)
N_cpus = cpu_count()

pool = get_context("fork").Pool(N_cpus)        # M1 chip -- allows multiprocessing

        ## this is the fuction for calling the ensamble
ensemble = EnsembleSampler(
        nwalkers,  # number of walkers defined 
        ndims,  # dimension of the problem 
        log_like_fn, # likelihood function
        priors, 
        args=[fft_data ,  df,freqs,h],  # data , sampling frequency, frequencies used, filter
        tempering_kwargs=tempering_kwargs, 
        moves=moves , # set to true if RJ is used
        rj_moves=False, # set to true if RJ is used
        provide_groups=False, # set to true if RJ is used
        nleaves_max=nleaves_max,
        nleaves_min=nleaves_min,
        branch_names=branch_names,
        pool = pool,
        update_iterations=10, # to use the update function
        update_fn=update_fn,  # to use the Supdate function
        stopping_fn=stop,  # to use the stopping function
        stopping_iterations=500,
        nbranches=1, 
        vectorize=True, # vectorized likelihood
        backend=(fp))


nsteps =20000
thin_by=1
burn=0
    

print('start')

log_prior = ensemble.compute_log_prior(coords, inds=inds)
log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]
start_state = State(coords, inds=inds , log_like=log_like, log_prior=log_prior)
out = ensemble.run_mcmc(start_state, nsteps, burn=burn, progress=True, thin_by=thin_by)