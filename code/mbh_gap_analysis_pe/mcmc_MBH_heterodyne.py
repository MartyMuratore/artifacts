"""
MCMC Parameter Estimation for Massive Black Hole Binaries with LISA

This script performs Bayesian parameter estimation for MBH binaries using:
- Heterodyned likelihood for fast MCMC sampling (~100-1000x speedup)
- Gap-aware analysis with time-domain masking
- Fisher matrix informed priors
- Parallel tempering support

Key Features:
- Centralized configuration via config.py
- Modular design using mbh_analysis package
- Support for GPU acceleration (CuPy)
- Flexible gap scenarios (masks, windows)

Usage:
    python mcmc_MBH_heterodyne.py

Configuration:
    Edit config.py or override specific settings in this file

Author: Ollie Burke
Date: October 2025
"""

import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from bbhx.waveformbuild import BBHWaveformFD

# Import configuration
from config import get_default_config

# Import from mbh_analysis package (new modular structure)
from mbh_analysis import (
    MBH_f,                      # Waveform generation
    GapLikelihood,              # Heterodyned likelihood (self-contained)
    inner_prod,                 # Noise-weighted inner product
    generate_colored_noise,     # Colored noise generation
    load_psd_from_file,         # PSD loading/interpolation
    create_gap_window,          # Gap windowing functions
    build_fish_matrix           # Fisher matrix computation
)

from lisatools.utils.constants import YRSID_SI

from eryn.ensemble import EnsembleSampler
from eryn.moves import StretchMove
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.backends import HDFBackend


# ================ LOAD CONFIGURATION =============
# Create configuration object - modify parameters here or in config.py
cfg = get_default_config()

# Optional: Override specific settings
# cfg.use_heterodyne = True
# cfg.n_iterations = 1000
# cfg.use_gpu = False

# Print configuration summary
cfg.print_summary()

# Extract commonly used variables for backward compatibility
PLOT_WAVEFORM = cfg.plot_waveform
CHECK_SNR = cfg.check_snr
NO_MASK = cfg.no_mask
MASK = cfg.mask
WINDOW = cfg.window
USE_HETERODYNE = cfg.use_heterodyne

noise_direc = cfg.noise_dir
mcmc_direc = cfg.mcmc_dir
fisher_direc = cfg.fisher_dir

N_channels = cfg.n_channels
channel = cfg.channels
sens_fn_calls = cfg.sens_fn_calls

use_gpu = cfg.use_gpu
xp = cfg.xp

seed_number = cfg.seed_number

##======================Likelihood Setup (HETERODYNED)=====================

# We'll define the likelihood function later, after setting up the GapLikelihood object

amp_phase_kwargs = cfg.amp_phase_kwargs
response_kwargs = cfg.response_kwargs
wave_gen = BBHWaveformFD(amp_phase_kwargs=amp_phase_kwargs, response_kwargs=response_kwargs)

# Get injection parameters from config
m1 = cfg.m1
m2 = cfg.m2
a1 = cfg.a1
a2 = cfg.a2
inc = cfg.inc
dist_Gpc = cfg.dist_Gpc

beta = cfg.beta
lam = cfg.lam
psi = cfg.psi

t_ref = cfg.t_ref
f_ref = cfg.f_ref
phi_ref = cfg.phi_ref

M = cfg.M
q = cfg.q

modes = cfg.modes

params = cfg.params
N_params = cfg.n_params

delta_t = cfg.delta_t
T_obs = cfg.t_obs
delta_f = cfg.delta_f

time = cfg.time
N = cfg.N
freq = cfg.freq

kwargs = cfg.waveform_kwargs

MBH_AE = MBH_f(wave_gen, *params, **kwargs)

##===========================Convert into time-domain ===================
# BBHx waveforms are in Hz^-1, divide by delta_t before IRFFT
MBH_AE_t = xp.asarray([xp.fft.irfft(MBH_AE[k] / delta_t) for k in range(N_channels)])
sim_t = xp.arange(len(MBH_AE_t[0])) * delta_t
###========================SET UP GAP SITUATION ===========================
# Get gap configuration from config
gap_centers = cfg.gap_centers
gap_widths = cfg.gap_widths
lobe_widths = cfg.lobe_widths

if MASK:
    gap_window_array = create_gap_window(sim_t, gap_centers, gap_widths, lobe_widths=0.0, use_gpu=use_gpu)
elif WINDOW:
    gap_window_array = create_gap_window(sim_t, gap_centers, gap_widths, lobe_widths=lobe_widths, use_gpu=use_gpu)
else:
    gap_window_array = np.ones(len(sim_t))

# =============== PLACE THE GAP WINDOW ONTO THE WAVEFORM ==============
MBH_AE_t*=gap_window_array
# Multiply by delta_t after RFFT to get back to Hz^-1
MBH_AE_f = xp.asarray([xp.fft.rfft(MBH_AE_t[k]) * delta_t for k in range(N_channels)])
# Define PSDs
PSD_filename = cfg.psd_filename

PSD_AE_interp = load_psd_from_file(cfg.get_full_path(PSD_filename, 'noise'), xp=xp)

freq_np = xp.asarray(freq)

PSD_AE = PSD_AE_interp(freq_np)

kwargs['PSD'] = PSD_AE
SNR2_AET = xp.asarray([inner_prod(MBH_AE_f[i],MBH_AE_f[i],PSD_AE[i], delta_f, xp=xp) for i in range(N_channels)])

for i in range(N_channels):
    print("For channel {}, we observe SNR = {}".format(channel,SNR2_AET[i]**(1/2)))

print("Total SNR for A, E, T is given by", xp.sum(SNR2_AET)**(1/2))

# Compute Variance and build noise realisation

# ======================== GENERATE NOISE REALISATION ====================
N = len(sim_t)
# With BBHx normalization, variance = PSD / (4*delta_f)
variance_noise_AET = [PSD_AE[k] / (4*delta_f) for k in range(N_channels)]

if CHECK_SNR:
    SNR_vec = []
    for j in tqdm(range(0,100)):
        noise_f_AE = generate_colored_noise(variance_noise_AET, delta_t, seed=j, window_function=gap_window_array, return_time_domain=False)

        num = xp.sum(xp.asarray([inner_prod(MBH_AE_f[i] + noise_f_AE[i], MBH_AE_f[i], PSD_AE[i], delta_f, xp=xp) for i in range(N_channels)]))**2
        denom = xp.sum(xp.asarray([inner_prod(MBH_AE_f[i], MBH_AE_f[i], PSD_AE[i], delta_f, xp=xp) for i in range(N_channels)]))

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
    noise_t_AE = generate_colored_noise(variance_noise_AET, delta_t, seed=0, window_function=gap_window_array, return_time_domain=True)
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
noise_f_AE = generate_colored_noise(variance_noise_AET, delta_t, seed=0, window_function=gap_window_array, return_time_domain=False)
data_f_AE = MBH_AE_f + noise_f_AE

##===========================MCMC Settings============================

# Get MCMC settings from config
iterations = cfg.n_iterations
burnin = cfg.burnin
nwalkers = cfg.n_walkers
ntemps = cfg.n_temps

tempering_kwargs = dict(ntemps=ntemps)

# Build Fisher matrix
Fish_Matrix = build_fish_matrix(wave_gen, M, q, a1, a2, inc, dist_Gpc,
                                phi_ref, lam, beta, psi, t_ref,
                                window_func=gap_window_array, **kwargs)
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
    'param_names': cfg.param_names,
    'delta_f': delta_f,
    'freq_range': [float(freq[0]), float(freq[-1])],
    'modes': modes
}
np.save(cfg.fisher_filepath, fisher_results, allow_pickle=True)

Delta_params = np.diag(Cov_Matrix)**(1/2)
d = cfg.start_scatter_factor

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

n = cfg.fisher_prior_scale  # Prior width = n * Fisher uncertainty

# Fisher uncertainties
Delta_M = Delta_params[0]
Delta_q = Delta_params[1]
Delta_a1 = Delta_params[2]
Delta_a2 = Delta_params[3]
Delta_inc = Delta_params[4]
Delta_dist = Delta_params[5]
Delta_phi_ref = Delta_params[6]
Delta_lam = Delta_params[7]
Delta_beta = Delta_params[8]
Delta_psi = Delta_params[9]
Delta_t_ref = Delta_params[10]

priors_in = {
    0: uniform_dist(M - n*Delta_M, M + n*Delta_M),           # Total mass M
    1: uniform_dist(q - n*Delta_q, q + n*Delta_q),           # Mass ratio q
    2: uniform_dist(a1 - n*Delta_a1, a1 + n*Delta_a1),       # Primary spin a1
    3: uniform_dist(a2 - n*Delta_a2, a2 + n*Delta_a2),       # Secondary spin a2
    4: uniform_dist(inc - n*Delta_inc, inc + n*Delta_inc),   # Inclination
    5: uniform_dist(dist_Gpc - n*Delta_dist, dist_Gpc + n*Delta_dist),  # Distance
    6: uniform_dist(phi_ref - n*Delta_phi_ref, phi_ref + n*Delta_phi_ref),  # Phase
    7: uniform_dist(lam - n*Delta_lam, lam + n*Delta_lam),   # Longitude
    8: uniform_dist(beta - n*Delta_beta, beta + n*Delta_beta),  # Latitude
    9: uniform_dist(psi - n*Delta_psi, psi + n*Delta_psi),   # Polarization
    10: uniform_dist(t_ref - n*Delta_t_ref, t_ref + n*Delta_t_ref)  # Reference time
}

priors = ProbDistContainer(priors_in, use_cupy=use_gpu)

# =================== SET UP PROPOSAL ==================

moves_stretch = StretchMove(a=cfg.stretch_move_a, use_gpu=use_gpu)

##======================SET UP HETERODYNED LIKELIHOOD=====================

print("\n" + "="*80)
print("SETTING UP LIKELIHOOD")
print("="*80)

if USE_HETERODYNE:
    print("\nUsing HETERODYNED likelihood (fast, ~100-1000x speedup)")
    print("Initializing GapLikelihood with heterodyned mode...")

    # Create the heterodyned likelihood object
    likelihood_obj = GapLikelihood(
        wave_gen=wave_gen,
        data_channels=data_f_AE,
        data_freqs=freq,
        gap_window_array=gap_window_array,
        delta_t=delta_t,
        delta_f=delta_f,
        N=N,
        PSD_interp=PSD_AE_interp,
        reference_params=params,  # Use injection parameters as reference
        f_ref=f_ref,
        modes=modes,
        N_channels=N_channels,
        use_heterodyned=True,  # Start in heterodyned mode
        use_gpu=use_gpu
    )

    # Define likelihood function
    def llike(params_in):
        """Heterodyned log-likelihood function"""
        return likelihood_obj(params_in)

    print("✓ Heterodyned likelihood initialized successfully")
    print(f"  Reference SNR: {xp.sum(SNR2_AET)**(1/2):.2f}")
    print(f"  Sparse grid size: 128 points (vs {len(freq)} dense grid)")
    print(f"  Expected speedup: ~{len(freq)//128}x")

else:
    print("\nUsing DIRECT likelihood (slow but exact)")
    print("Note: For production runs, set USE_HETERODYNE = True for ~100-1000x speedup")

    # Direct likelihood (original implementation)
    def llike(params_in):
        """
        Whittle-based log-likelhood function used for the parameter inference
        """
        waveform_prop_f_AE = MBH_f(wave_gen, *params_in, **kwargs)
        # BBHx waveforms are in Hz^-1, need to divide by delta_t before IRFFT
        waveform_prop_t_AE = xp.asarray([gap_window_array*xp.fft.irfft(waveform_prop_f_AE[k]/delta_t) for k in range(N_channels)])
        # Multiply by delta_t after RFFT to get back to Hz^-1
        waveform_prop_f_AE = xp.asarray([xp.fft.rfft(waveform_prop_t_AE[k])*delta_t for k in range(N_channels)])

        diff_f_AE = [data_f_AE[k] - waveform_prop_f_AE[k] for k in range(N_channels)]
        inn_prod = xp.asarray([inner_prod(diff_f_AE[k],diff_f_AE[k],PSD_AE[k], delta_f, xp=xp) for k in range(N_channels)])
        return(-0.5 * xp.sum(inn_prod))

    print("✓ Direct likelihood function defined")

# Posterior function (same for both)
def lpost(params_in):
    '''Compute log posterior'''
    log_prior = priors.logpdf(params_in)
    if xp.isinf(log_prior):
        print("Prior returns -inf")
        return -np.inf
    else:
        return llike(params_in) + log_prior

print("="*80)

# NOTE: To switch between heterodyned and direct likelihood:
#   1. Set USE_HETERODYNE = False in config.py (or above)
#   2. Restart the script
# The GapLikelihood class handles both modes automatically!

# Quick checks
if ntemps > 1:
    print("Value of starting log-likelihood points", llike(start[0][0]))
    if np.isinf(sum(priors.logpdf(np.asarray(start[0])))):
        print("You are outside the prior range, you fucked up")
        quit()
else:
    print("Value of starting log-likelihood points", llike(start[0]))

breakpoint()

# Get output filepath from config
fp = cfg.output_filepath
backend = HDFBackend(fp)

# Set up multiprocessing pool (CPU only)
if not use_gpu and cfg.use_multiprocessing:
    from multiprocessing import get_context, cpu_count
    N_cpus = cpu_count()
    pool = get_context("fork").Pool(N_cpus)
else:
    pool = None

breakpoint()

# Set up ensemble sampler
ensemble = EnsembleSampler(
    nwalkers,
    ndim,
    llike,
    priors,
    pool=pool,
    backend=backend,
    tempering_kwargs=tempering_kwargs,
    moves=moves_stretch
)

# Reset backend if configured (WARNING: Deletes existing MCMC file!)
Reset_Backend = cfg.reset_backend
if Reset_Backend:
    print(f"\nWARNING: Resetting backend - deleting {fp}")
    if os.path.exists(fp):
        os.remove(fp)
    backend = HDFBackend(fp)
    ensemble = EnsembleSampler(
        nwalkers,
        ndim,
        llike,
        priors,
        pool=pool,
        backend=backend,
        tempering_kwargs=tempering_kwargs,
        moves=moves_stretch
    )
else:
    print(f"\nContinuing from previous run: {fp}")
    start = backend.get_last_sample()

# Run MCMC
print(f"\nStarting MCMC run: {iterations} iterations, {nwalkers} walkers")
out = ensemble.run_mcmc(start, iterations, progress=True)
print("\nMCMC run complete!")
##===========================MCMC Settings (change this)============================
