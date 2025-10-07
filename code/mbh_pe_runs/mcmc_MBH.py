import os
import sys
import numpy as np
from tqdm import tqdm as tqdm
import matplotlib.pyplot as plt
from bbhx.waveformbuild import BBHWaveformFD
from tqdm import tqdm
from mbh_utils import MBH_f, build_fish_matrix 
from noise_utils import load_psd_from_file, inner_prod, generate_colored_noise, pad_to_length
from window_func import create_gap_window, gap_definitions, taper_defs, include_planned, include_unplanned, planned_seed, unplanned_seed

from lisatools.utils.constants import YRSID_SI
from lisaglitch import GapMaskGenerator
from lisagap import GapWindowGenerator


from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend


# ================ SETTINGS PRIOR TO SIMULATION =============
PLOT_WAVEFORM =True # Whether we decide to plot waveform and noise
GPU_DIRECTORY = False
OLLIE_DIRECTORY = True
NO_MASK = True
CHECK_SNR = True
MASK = False
WINDOW = False

if OLLIE_DIRECTORY:
    noise_direc = "/Users/ollie.burke/Documents/Work/Code/spritz_challenge/code/mbh_pe_runs/" 
    mcmc_direc = "/Users/ollie.burke/Documents/Work/Code/spritz_challenge/code/mbh_pe_runs/data_mcmc_simulations/"
    fisher_direc = "/Users/ollie.burke/Documents/Work/Code/spritz_challenge/code/mbh_pe_runs/fisher_results/"

N_channels = 2
channel = ["A","E"]
sens_fn_calls = ["noisepsd_AE","noisepsd_AE"]

use_gpu = False
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

seed_number = 0
##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Whittle-based log-likelhood function used for the parameter inference
    """

    waveform_prop_f_AE = MBH_f(wave_gen,*params, **kwargs)
    waveform_prop_t_AE = xp.asarray([gap_window_array*xp.fft.irfft(waveform_prop_f_AE[k]) for k in range(N_channels)])
    waveform_prop_f_AE = xp.asarray([xp.fft.rfft(waveform_prop_t_AE[k]) for k in range(N_channels)])

    diff_f_AE = [data_f_AE[k] - waveform_prop_f_AE[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AE[k],diff_f_AE[k],PSD_AE[k],N, delta_t) for k in range(N_channels)])
    return(-0.5 * xp.sum(inn_prod))

def lpost(params):
    '''
    Compute log posterior
    '''
    if xp.isinf(lprior(params)):
        print("Prior returns -\infty")
        return -np.inf
    else:
        return llike(params)

amp_phase_kwargs = dict(run_phenomd=False)

response_kwargs = dict(TDItag="AET", tdi2=True)
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=response_kwargs)

m1 = 1323277.47932  #/(1 + 1.73941)
m2 =  612485.50602999  #/(1 + 1.73941)
a1 = 0.747377 # spin 1
a2 =  0.8388   # spin 2
inc = np.pi/3 #inclination
dist_Gpc = 36.90249521628649 # luminosity distance

beta = -0.30300442294174235  # ecliptic latitude
lam =   1.2925183861048521 # ecliptic longitude
psi = np.pi/6 # polarization angle

t_ref = 2627744.9218792617
f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2 # phase at f_ref

M = (m1 + m2) 
q = m2 / m1  # m2 less than m1 

modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]) 
N_params = len(params)

delta_t = 5.0
T_obs = 1*31*24*3600  # 2 months of data + padding on either side
delta_f = 1/T_obs 

time = np.arange(0,T_obs,delta_t)
N = len(time)
# freq = xp.arange(1e-5,1e-1,delta_f)
freq = xp.arange(1e-5,1/(2*delta_t),delta_f)

kwargs = {"freq" : freq,
          "delta_t": delta_t,
          "delta_f" : delta_f,
          "f_ref" : f_ref,
          "modes" : modes}

MBH_AE = MBH_f(wave_gen, *params, **kwargs)

##===========================Convert into time-domain ===================
MBH_AE_t = xp.asarray([xp.fft.irfft(MBH_AE[k]) for k in range(N_channels)])
sim_t = xp.arange(len(MBH_AE_t[0])) * delta_t
###========================SET UP GAP SITUATION ===========================
# Initialise the class with simulation properties and whether or not to treat gaps with
# nans or not. 
# Create 3 gaps with different widths
gap_centers = [2.600e6, 0*2.6265e6, 2.633e6]  # 10, 30, 50 days
gap_widths = [7*3600, 0*0.5*3600, 1*3600]  # 3hr, 2hr, 4hr gaps

if MASK:
    gap_window_array= create_gap_window(sim_t, gap_centers, gap_widths,lobe_widths = 0.0, use_gpu=False)
elif WINDOW: 
    lobe_widths = 5*60  # 5-minute tapers
    gap_window_array= create_gap_window(sim_t, gap_centers, gap_widths, lobe_widths = lobe_widths, use_gpu=False)
else:
    gap_window_array =np.ones(len(sim_t))

# =============== PLACE THE GAP WINDOW ONTO THE WAVEFORM ==============
MBH_AE_t*=gap_window_array 
MBH_AE_f = xp.asarray([xp.fft.rfft(MBH_AE_t[k]) for k in range(N_channels)])
# Define PSDs
# First, write PSD to a file.

PSD_filename = "tdi2_AE_w_background.npy"

PSD_AE_interp = load_psd_from_file(noise_direc + PSD_filename, xp=xp)

freq_np = xp.asarray(freq)

PSD_AE = PSD_AE_interp(freq_np)

kwargs['PSD'] = PSD_AE
SNR2_AET = xp.asarray([inner_prod(MBH_AE_f[i],MBH_AE_f[i],PSD_AE[i],N, delta_t) for i in range(N_channels)])

for i in range(N_channels):
    print("For channel {}, we observe SNR = {}".format(channel,SNR2_AET[i]**(1/2)))

print("Total SNR for A, E, T is given by", xp.sum(SNR2_AET)**(1/2))

# Compute Variance and build noise realisation

# ======================== GENERATE NOISE REALISATION ====================
N = len(sim_t)
variance_noise_AET = [N * PSD_AE[k] / (4*delta_t) for k in range(N_channels)]

if CHECK_SNR:
    SNR_vec = []
    for j in tqdm(range(0,100)):
        noise_f_AE = generate_colored_noise(variance_noise_AET, seed=j, window_function=gap_window_array, return_time_domain=False)

        num = xp.sum(xp.asarray([inner_prod(MBH_AE_f[i] + noise_f_AE[i], MBH_AE_f[i], PSD_AE[i], N, delta_t) for i in range(N_channels)]))**2
        denom = xp.sum(xp.asarray([inner_prod(MBH_AE_f[i], MBH_AE_f[i], PSD_AE[i], N, delta_t) for i in range(N_channels)]))

        SNRs = xp.sqrt(num/denom)
        SNR_vec.append(SNRs)
        print(SNRs)

    plt.hist(SNR_vec, bins = 20)
    plt.xlabel(r'SNR')
    plt.ylabel(r'Histogram')
    plt.axvline(x = xp.sum(SNR2_AET)**(1/2), c = 'red', linestyle = 'dashed', label = 'Expected SNR')
    plt.show()
# =============== Plot our waveform =================
if PLOT_WAVEFORM:
    noise_t_AE = generate_colored_noise(variance_noise_AET, seed=0, window_function=gap_window_array, return_time_domain=True)
    plt.figure(figsize=(10, 6))
    plt.plot(sim_t, xp.asarray(noise_t_AE[0]) if use_gpu else MBH_AE_t[0] + noise_t_AE[0], label = 'Data TDI2')
    plt.plot(sim_t, xp.asarray(MBH_AE_t[0]) if use_gpu else MBH_AE_t[0], label = 'BBHx', c = 'red', alpha = 0.4) 
    window_rescaled = xp.asarray(np.max(MBH_AE_t[0]))*gap_window_array if use_gpu else np.max(MBH_AE_t[0])*gap_window_array
    plt.plot(sim_t, window_rescaled, label = 'BBHx (Windowed)')
    plt.axvline(x = t_ref, c = 'red', linestyle = 'dashed', label = 't_ref')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('MBH Waveform - Channel A (Time Domain)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot characteristic strain vs PSD
    plt.figure(figsize=(10, 6))
    # Compute characteristic strain: h_c(f) = 2|h(f)| * sqrt(f)
    char_strain_A = 2 * xp.abs(MBH_AE_f[0]) * freq_np
    char_strain_A = xp.asnumpy(char_strain_A) if use_gpu else char_strain_A
    
    # Convert PSD to strain sensitivity: sqrt(f * S_n(f))
    strain_sensitivity_A = xp.sqrt(freq_np * PSD_AE[0])
    strain_sensitivity_A = xp.asnumpy(strain_sensitivity_A) if use_gpu else strain_sensitivity_A
    
    freq_plot = xp.asnumpy(freq_np) if use_gpu else freq_np
    
    plt.loglog(freq_plot, char_strain_A, label='Characteristic Strain (Channel A)', linewidth=2)
    plt.loglog(freq_plot, strain_sensitivity_A, label='Strain Sensitivity (sqrt(f*PSD))', linewidth=2, alpha=0.7)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Strain')
    plt.title('Characteristic Strain vs Noise PSD')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    plt.xlim(freq_plot[freq_plot > 0].min(), freq_plot.max())
    plt.tight_layout()
    plt.show()




# Compute noise in frequency domain
noise_f_AE = generate_colored_noise(variance_noise_AET, seed=0, window_function=gap_window_array, return_time_domain=False)
data_f_AE = MBH_AE_f + noise_f_AE

##===========================MCMC Settings============================

iterations = 30000  # The number of steps to run of each walker
burnin = 0 # I always set burnin when I analyse my samples
nwalkers = 50  #50 #members of the ensemble, like number of chains

# USING ntemps = 5 for the Kerr inj AAK rec runs
ntemps = 1             # Number of temperatures used for parallel tempering scheme.
                       # Each group of walkers (equal to nwalkers) is assigned a temperature from T = 1, ... , ntemps.

tempering_kwargs=dict(ntemps=ntemps)  # Sampler requires the number of temperatures as a dictionary

# n = 0
Fish_Matrix = build_fish_matrix(wave_gen, M, q, a1, a2, inc, dist_Gpc, 
                                phi_ref, lam,beta,psi,t_ref, 
                                window_func = gap_window_array,**kwargs)
Cov_Matrix = np.linalg.inv(Fish_Matrix)

# Save Fisher matrix results with metadata
fisher_results = {
    'Cov_Matrix': Cov_Matrix,
    'Fish_Matrix': Fish_Matrix,
    'SNR_per_channel': xp.asnumpy(SNR2_AET**(1/2)) if use_gpu else SNR2_AET**(1/2),
    'SNR_total': float(xp.asnumpy(xp.sum(SNR2_AET)**(1/2))) if use_gpu else float(xp.sum(SNR2_AET)**(1/2)),
    'PSD_filename': PSD_filename,
    'true_params': {
        'M': M,
        'q': q,
        'a1': a1,
        'a2': a2,
        'inc': inc,
        'dist_Gpc': dist_Gpc,
        'phi_ref': phi_ref,
        'lam': lam,
        'beta': beta,
        'psi': psi,
        't_ref': t_ref
    },
    'param_names': ['M', 'q', 'a1', 'a2', 'inc', 'dist_Gpc', 'phi_ref', 'lam', 'beta', 'psi', 't_ref'],
    'delta_f': delta_f,
    'freq_range': [float(freq[0]), float(freq[-1])],
    'modes': modes
}
np.save(fisher_direc + 'fisher_results.npy', fisher_results, allow_pickle=True)

Delta_params = np.diag(Cov_Matrix)**(1/2)
d = 0.01 

#here we should be shifting by the *relative* error! 

start_M = M + d * Delta_params[0] * np.random.randn(nwalkers,1)   
start_q = q + d * Delta_params[1] * np.random.randn(nwalkers,1)
start_a1 = a1 + d * Delta_params[2] * np.random.randn(nwalkers,1)
start_a2 = a2 + d * Delta_params[3] * np.random.randn(nwalkers, 1)
start_inc = inc + d * Delta_params[4] * np.random.randn(nwalkers, 1)
start_dist_Gpc = dist_Gpc + d * Delta_params[5] * np.random.randn(nwalkers, 1)

start_phi_ref = phi_ref + d * Delta_params[6] * np.random.randn(nwalkers,1)
start_lam = lam + d * Delta_params[7] * np.random.randn(nwalkers,1)
start_beta = beta + d * Delta_params[8] * np.random.randn(nwalkers,1)
start_psi = psi + d * Delta_params[9] * np.random.randn(nwalkers,1)

start_t_ref = t_ref + d * Delta_params[10] * np.random.randn(nwalkers, 1)

start = np.hstack((start_M,start_q, start_a1, start_a2, start_inc, start_dist_Gpc, start_phi_ref, start_lam, start_beta, start_psi, start_t_ref))

if ntemps > 1:
    # If we decide to use parallel tempering, we fall into this if statement. We assign each *group* of walkers
    # an associated temperature. We take the original starting values and "stack" them on top of each other. 
    start = np.tile(start,(ntemps,1,1))

if np.size(start.shape) == 1:
    start = start.reshape(start.shape[-1], 1)
    ndim = 1
else:
    ndim = start.shape[-1]
# ================= SET UP PRIORS ========================


n = 10  # size of prior

# Tighter priors for high SNR
Delta_M = Delta_params[0]     # 1% uncertainty on total mass M
Delta_q = Delta_params[1]          # Tighter uncertainty on mass ratio q
Delta_a1 = Delta_params[2]        # Tighter spin uncertainty
Delta_a2 = Delta_params[3]        # Tighter spin uncertainty
Delta_inc = Delta_params[4]
Delta_dist = Delta_params[5]

Delta_phi_ref = Delta_params[6]
Delta_lam = Delta_params[7]
Delta_beta = Delta_params[8]
Delta_psi = Delta_params[9]
Delta_t_ref = Delta_params[10]

priors_in = {
    # Intrinsic parameters
    # Total Mass M (2e6 Msun)
        0: uniform_dist(M - n*Delta_M, M + n*Delta_M),  # Mass ratio q (>=1)
        1: uniform_dist(q - n*Delta_q, q + n*Delta_q),  # Mass ratio q (>=1)
        2: uniform_dist(a1 - n*Delta_a1, a1 + n*Delta_a1),  # Primary spin a1
        3: uniform_dist(a2 - n*Delta_a2, a2 + n*Delta_a2),  # Secondary spin a2
        4: uniform_dist(inc - n*Delta_inc, inc + n*Delta_inc),  # Inclination
        5: uniform_dist(dist_Gpc - n*Delta_dist, dist_Gpc + n*Delta_dist),  # Distance (Gpc) 
        # Extrinsic parameters
        6: uniform_dist(phi_ref - n*Delta_phi_ref, phi_ref + n*Delta_phi_ref),  # Reference phase
        7: uniform_dist(lam - n*Delta_lam, lam + n*Delta_lam),  # Ecliptic longitude lambda
        8: uniform_dist(beta - n*Delta_beta, beta + n*Delta_beta),  # Ecliptic latitude beta
        9: uniform_dist(psi - n*Delta_psi, psi + n*Delta_psi),  # Polarization angle psi
        10: uniform_dist(t_ref - n*Delta_t_ref, t_ref + n*Delta_t_ref)  # Reference time
}

priors = ProbDistContainer(priors_in, use_cupy = False)   # Set up priors so they can be used with the sampler.

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=2.0, use_gpu=use_gpu)

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0])) 
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0])) 
breakpoint()
fp = mcmc_direc + f"MBH_HMs_case_1_tdi2_SNR_1137_AE_w_noise_no_mask_seed_{seed_number}.h5"
backend = HDFBackend(fp)

if use_gpu == False:
    from multiprocessing import (get_context,cpu_count)
    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)        # M1 chip -- allows multiprocessing
else:
    pool = None
ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            pool = pool,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
Reset_Backend = True # NOTE: CAREFUL HERE. ONLY TO USE IF WE RESTART RUNS!!!!
if Reset_Backend:
    os.remove(fp) # Manually get rid of backend
    backend = HDFBackend(fp) # Set up new backend
    ensemble = EnsembleSampler(
                            nwalkers,          
                            ndim,
                            llike,
                            priors,
                            pool = pool,
                            backend = backend,                 # Store samples to a .h5 file
                            tempering_kwargs=tempering_kwargs,  # Allow tempering!
                            moves = moves_stretch
                            )
else:
    start = backend.get_last_sample() # Start from last sample
out = ensemble.run_mcmc(start, iterations, progress=True)  # Run the sampler
##===========================MCMC Settings (change this)============================
