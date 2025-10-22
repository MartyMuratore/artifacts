### import python packages packages

import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import csd


### import packages for signal processing and covariance matrices computations

from scipy.signal import welch
from scipy.signal import butter, filtfilt, freqz

### import Eryn 

from eryn.ensemble import EnsembleSampler
from eryn.state import State
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.utils import  SearchConvergeStopping, Stopping
from eryn.backends import HDFBackend
from eryn.utils import SearchConvergeStopping
from eryn.moves import StretchMove

### import lisatools

import sys
sys.path.append("../")


DELETE_BACKEND = True
coords_prior = False


use_gpu = False

if use_gpu is False:
    xp = np

### setting number and name of branches
branch_names = ["noise"]

### setting the dimensionality of the parameter to fix
ndims = {"noise": 1} 

### setting the Reversible Jump on the glitches allowing between 1 and 0 glitches
nleaves_max = {"noise": 1} 
nleaves_min = { "noise": 1} 

### setting the number of walkers to use and the number of temperatures 

nwalkers = 20
ntemps = 1
# Tmax = np.inf
Tmax = 1.0
tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax) # here the maximum temperature is the to infinite so that we ensure sampling the priors ( see https://arxiv.org/abs/2303.02164 )
                        
   
parameter_noise_amplitude = [2.4e-15, 1.42e-12,3.32e-12,6.35e-12,3.0e-12,3.0e-12]
average_armlength = 8.322688660167833
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
                        T = average_armlength):

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
    readout_tmi_noise = readout_tmi**2 *(1+ (2e-3 / f)**4)


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
    
 
    return total_noise_XX,total_noise_XY

####### ---------- defining the GAP mask ---------- ###

# maskgapX = np.where((data_tdi_X==0),0,1).astype(np.int32)
# maskgapY = np.where((data_tdi_Y==0),0,1).astype(np.int32)
# maskgapZ = np.where((data_tdi_Z==0),0,1).astype(np.int32)

########## ------------ comparing noise data with models NO FILTER-------- ####

freqs = np.fft.rfftfreq(len(time), dt)  # fs =1/dt
df = freqs[1] - freqs[0]  # 1 / (dt * len(t_in))

freq_welch_f, psd_data_XX= welch(data_tdi_X, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//120, noverlap=50)
f_coh, Cxy_data = csd(data_tdi_X, data_tdi_Y, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//12, noverlap=50)

plt.figure()
plt.loglog(freq_welch_f[1:],np.sqrt(noise_models_correlation_spritz(freq_welch_f)[0])[1:],'-',alpha=1,label = 'PSD data')
plt.loglog(freq_welch_f[1:],np.sqrt(psd_data_XX)[1:],'-',alpha=1,label = 'PSD model')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
plt.legend()
plt.xlim([2.8e-5,freqs[-1]])
plt.savefig("plots/data_vs_model.png")
plt.close()

plt.figure()
plt.loglog(f_coh[1:],np.abs(noise_models_correlation_spritz(f_coh)[1])[1:],'-',alpha=1,label = 'CSD model')
plt.loglog(f_coh[1:],np.abs(Cxy_data)[1:],'-',alpha=1,label = 'CSD data')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
plt.xlim([2.8e-5,freqs[-1]])
plt.legend()
plt.savefig("plots/data_vs_model_csd.png")
plt.close()

########## ------------ comparing noise data with models WITH FILTER-------- ####

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
freqs = np.fft.rfftfreq(len(X_data_filtered), dt)  # fs =1/dt

## --------------------------  Get the filter's frequency response -------------  ##

freq_welch_f, psd_data_XX_filtered= welch(Y_data_filtered, fs=1/dt, window='boxcar', nperseg=len(X_data_filtered)//120, noverlap=50)
f_coh, Cxy_data_filtered = csd(Y_data_filtered,Z_data_filtered, fs=1/dt, window='boxcar', nperseg=len(data_tdi_X)//120, noverlap=50)

####### ------------- Apply the filter in the frequency domain to the unfiltered noise PSD ------------ ######

_, h = freqz(b, a, worN=len(freqs), fs=1/dt)

Sxx_unfiltered = noise_models_correlation_spritz(freqs)[0]
Sxy_unfiltered = noise_models_correlation_spritz(freqs)[1] 

Sxx_filtered = Sxx_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response
Sxy_filtered = Sxy_unfiltered * np.abs(h)**4  # Squared magnitude of the filter response

## ----------- Plots model vs data with filter -------##

plt.figure()
plt.loglog(freqs[1:],np.sqrt(Sxx_filtered)[1:],'-',alpha=1,label = 'PSD model')
plt.loglog(freq_welch_f[1:],np.sqrt(psd_data_XX_filtered[1:]),'-',alpha=1,label = 'PSD data filtered')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
plt.xlim([2.5e-5,2.5e-2])
plt.legend()
plt.savefig("plots/data_vs_model.png")
plt.close()

plt.figure()
plt.loglog(f_coh[1:],np.abs(Cxy_data_filtered)[1:],'-',alpha=1,label = 'CSD data filtered')
plt.loglog(freqs[1:],np.abs(Sxy_filtered)[1:],'-',alpha=1,label = 'CSD model')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Spline example ')
plt.xlim([2.8e-5,2.5e-2])
plt.legend()
plt.savefig("plots/data_vs_model_cxy.png")
plt.close()


## ------------ Reducing the frequency range for the analysis to avoid dips -------------- ##

fmin = 2.5e-5
fmax = 2.5e-2
frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL
freqs_cut =  np.array(freqs[frequencymask])

Xnfft = np.fft.rfft(X_data_filtered ) * dt # TD glitch
Ynfft = np.fft.rfft(Y_data_filtered) * dt # TD glitch
Znfft = np.fft.rfft(Z_data_filtered) * dt # TD glitch

fft_data_XYZ = xp.array([Xnfft[frequencymask] ,Ynfft[frequencymask] ,Znfft[frequencymask]]) 
h = h[frequencymask]  


## -----------noise moves-----------##

moves = [(StretchMove(gibbs_sampling_setup ="noise",live_dangerously=False))]

starting_coords = np.zeros((ntemps * nwalkers, ndims['noise'] ))
coords = { "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"]))}


d2 = 1e-2  # 1% of the true value

starting_coords = parameter_noise_amplitude[0] + d2 *parameter_noise_amplitude[0]* np.abs(np.random.randn(nwalkers * ntemps, ndims['noise']))

coords["noise"] = starting_coords.reshape(ntemps, nwalkers, nleaves_max["noise"], ndims['noise'])


## -------------noise priors----------##

priors = {}

parameter_noise_amplitude = [2.4e-15, 1.42e-12, 3.32e-12, 6.35e-12, 3.0e-12, 3.0e-12]

priors_noise = {
    0: uniform_dist(1e-16, 1e-14),}
'''
1: uniform_dist(1.349e-12, 1.491e-12),
2: uniform_dist(3.154e-12, 3.486e-12),
3: uniform_dist(6.033e-12, 6.668e-12),
4: uniform_dist(2.85e-12, 3.15e-12),
5: uniform_dist(2.85e-12, 3.15e-12)
'''

priors['noise'] = ProbDistContainer(priors_noise)


### ---- coordinates definition for the noise from priors-------- ###

if coords_prior == True:
    coords = { "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"]))}

    coords["noise"] = priors["noise"].rvs(size=(ntemps, nwalkers,nleaves_max["noise"]))


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
        skipp =-int(500)

       
        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()[skipp:,0,:]))
        print("min last loglike",np.min(samp.get_log_like()[skipp:,0,:]))
     
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
            #print("rj \n",samp.rj_acceptance_fraction[0] )
            print("swap \n",samp.swap_acceptance_fraction)


        ####  ----------- noise posterios plots --------- #######

        likelihood = samp.get_log_like()[skipp:, 0,:]
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6)),
        # horizontal line
        plt.axhline(y=y_value, color='r', linestyle='--', linewidth=1.5, label=f'y = {y_value:.2e}')
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.xlabel('iter')
        plt.ylabel('Likelihood noise')
        plt.grid(True)
        plt.savefig("plots/noise_likelihood.png", dpi=300)
        plt.close()
        noise_sampler= samp.get_chain()["noise"][skipp:,0].reshape(-1,ndims['noise'])
        
        parameter_labels = [f'noise_amplitude_{j}' for j in range(0,ndims['noise'])]

        fig, axes = plt.subplots(1, 2, figsize=(15, 12))
        colors = ['#6495ed', '#ff6b6b', '#4ecdc4', '#95e1d3', '#f38181', 
                '#aa96da', '#fcbad3', '#a8d8ea', '#ffcb91', '#d4a5a5']

        # Flatten axes array for easier indexing
        axes_flat = axes.flatten()

        
        for i in range(ndims['noise']):
            axes_flat[i].plot(noise_sampler[:, i], 
                            color=colors[i], alpha=0.7, linewidth=0.5)
            axes_flat[i].set_xlabel('Iteration', fontsize=11)
            axes_flat[i].set_ylabel(parameter_labels[i], fontsize=11)
            axes_flat[i].grid(True, alpha=0.3)


        plt.tight_layout()
        plt.savefig("plots/noise_trace.png", dpi=300)
        plt.close()
       
   
        import corner
        import matplotlib.pyplot as plt
      
        fig = corner.corner(noise_sampler)#, truths=parameter_noise_amplitude[:ndims['noise']])
        fig.savefig("plots/noise_spritz.png", dpi=300)
        plt.close(fig)
 

def log_like_fn(x, data, freqs ,df,dt,filter_tf, subset=1):
    beta_params = x
    inds = np.arange(0, beta_params.shape[0] + 1, subset)
    if inds[-1] < beta_params.shape[0]:
        inds = np.concatenate([inds, np.array([beta_params.shape[0]])])
    logl_all = []
    
    for i in range(len(inds) - 1):
        start = int(inds[i])
        end = int(inds[i + 1])
        A = beta_params[start:end].squeeze()
        B, C, D, F, G = parameter_noise_amplitude[1:]

        psd_estimate = noise_models_correlation_spritz(
            freqs,
            acc_level=A,
            readout_tmi=B,
            readout_rfi=C,
            readout_isi=D,
            backlink_tmi=F,
            backlink_rfi=G,
            T=average_armlength,
        )

        # Covariance matrix components
        sigma_sq = psd_estimate[0] * np.abs(filter_tf)**4
        rho = psd_estimate[1] * np.abs(filter_tf)**4

        # Build covariance matrices for all frequencies
        csd_XY = np.array([
            [sigma_sq, rho, rho],
            [rho, sigma_sq, rho],
            [rho, rho, sigma_sq]
        ])  # shape (3, 3, n_freqs)


        '''
        plt.figure()
        plt.loglog(freqs,df*np.abs(data[0]*data[0]),'-',alpha=1,label='data')
        plt.loglog(freqs,np.abs(sigma_sq),'-',alpha=1,label='model')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Spline example ')
        plt.legend()
        plt.savefig("plots/estimated_model.png")
        plt.close()
        '''
        # Move frequency axis first for easier looping
        csd_XY_pinv = np.linalg.pinv(csd_XY.transpose(2, 0, 1))  # (n_freqs, 3, 3)

        # Quadratic term: data[n]^H Σ⁻¹ data[n]
        quad_terms = np.einsum('ni,nij,nj->n', np.conj(data.T), csd_XY_pinv, data.T)

        # Log-determinant term
        sign, logdet = np.linalg.slogdet(csd_XY.transpose(2, 0, 1) / dt)
        if np.any(sign <= 0):
            raise ValueError(f"Covariance not positive definite for A = {A:e}")

        # Compute log-likelihood
        
        loglike = -0.5*np.sum(logdet + 4* dt*quad_terms).real
    


        # Data components (use absolute value squared for complex data)

        
        logl = loglike[np.newaxis] if isinstance(loglike, (int, float, np.number)) else loglike
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
        args=[fft_data_XYZ , freqs_cut,df,dt,h], # data , sampling frequency, frequencies used, filter
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
'''

### this part is the likelihood we did together I comment it out as I am not super familiar with it

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
                freqs,
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
     
            cov_matrix_wo_filter = xp.asarray([
                    [psd_estimate[0], psd_estimate[1], psd_estimate[1]],
                    [psd_estimate[1], psd_estimate[0], psd_estimate[1]],
                    [psd_estimate[1], psd_estimate[1], psd_estimate[0]]])
    
   
            
            ## to account for the filter

            
            def compute_inverse_fully_symmetric(a, b):
                """
                Compute the inverse for the fully symmetric case:
                M = [[a, b, b],
                    [b, a, b],
                    [b, b, a]]
                
                Args:
                    a: Auto-PSD (diagonal vector, length n)
                    b: Cross-PSD (diagonal vector, length n) - same for all cross terms
                
                Returns:
                    B_diag, B_off, det: Diagonal and off-diagonal blocks of inverse, and determinant
                """
                # Compute determinant
                det = (a - b)**2 * (a + 2*b)
                
                # Inverse diagonal blocks
                B_diag = (a + b) / det
                
                # Inverse off-diagonal blocks
                B_off = -b / det
                
                return B_diag, B_off, det

            def compute_quadratic_form_symmetric(y_X, y_Y, y_Z, B_diag, B_off):
                """
                Compute y^T M^{-1} y for the symmetric case
                
                M⁻¹ has structure: diagonal blocks = B_diag, off-diagonal = B_off
                """
                # All diagonal terms use B_diag
                diag_terms = B_diag * (y_X**2 + y_Y**2 + y_Z**2)
                
                # All off-diagonal terms use B_off (factor of 2 for symmetry)
                off_diag_terms = 2 * B_off * (y_X*y_Y + y_X*y_Z + y_Y*y_Z)
                
                result = diag_terms + off_diag_terms
                
                return np.sum(result)

            def compute_log_det(det):
                """
                Compute log|M| from the determinant vector
                """
                return np.sum(np.log(det))

            
            def compute_trace_log(Matrix):
                return np.sum(np.log(np.diagonal(Matrix, axis1=0, axis2=1)))

            def likelihood_calculation_symmetric(y_X, y_Y, y_Z, a, b):
                """
                Complete likelihood calculation for fully symmetric case
                
                Args:
                    y_X, y_Y, y_Z: Data vectors (length n)
                    a: Auto-PSD (same for all channels)
                    b: Cross-PSD (same for all cross terms)
                
                Returns:
                    log_likelihood: -0.5 * (y^T M^{-1} y + log|M| + 3n*log(2π))
                """
       
       
                # Compute inverse blocks and determinant
                B_diag, B_off, det = compute_inverse_fully_symmetric(a,b)
                
                # Compute quadratic form
                quad_form = compute_quadratic_form_symmetric(y_X, y_Y, y_Z, B_diag, B_off)
                
                # Compute log determinant
                log_det = compute_log_det(det)

                trace_log= compute_trace_log(cov_matrix)
                
                # Likelihood (assuming Gaussian)
                log_likelihood1 = -0.5 * (quad_form + trace_log )
                log_likelihood2 = -0.5 * (quad_form + log_det )
                
                return log_likelihood1,log_likelihood2
            
            logl = likelihood_calculation_symmetric(data[0], data[1], data[2], psd_estimate[0] * np.abs(filter_tf)**4, psd_estimate[1] * np.abs(filter_tf)**4)[1]
            
                
    
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :])
                  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)
        if not isinstance(logl_out, np.ndarray):
            logl_out = xp.asnumpy(logl_out)
        return logl_out

'''