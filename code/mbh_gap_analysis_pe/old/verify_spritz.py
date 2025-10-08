#================= READ IN SPRITZ DATA ================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import h5py

from bbhx.waveformbuild import BBHWaveformFD

# Import configuration
from config import get_default_config

# Import utilities
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

# ================ LOAD CONFIGURATION =============
cfg = get_default_config(
    plot_waveform=True,
    window=True,
    mask=False,
    no_mask=False
)

# Extract commonly used variables
BH_directory = cfg.bh_directory
data = cfg.spritz_data_file

noise_direc = cfg.noise_dir
mcmc_direc = cfg.mcmc_dir
fisher_direc = cfg.fisher_dir

PLOT_WAVEFORM = cfg.plot_waveform
NO_MASK = cfg.no_mask
MASK = cfg.mask
WINDOW = cfg.window

run_direc = ""
N_channels = cfg.n_channels
channel = cfg.channels
sens_fn_calls = cfg.sens_fn_calls

use_gpu = cfg.use_gpu
xp = cfg.xp

seed_number = 0
##======================Likelihood and Posterior (change this)=====================

def llike(params):
    """
    Whittle-based log-likelhood function used for the parameter inference
    """

    waveform_prop_f_AE = MBH_f(wave_gen,*params, **kwargs)

    diff_f_AE = [data_f_AE[k] - waveform_prop_f_AE[k] for k in range(N_channels)]
    inn_prod = xp.asarray([inner_prod(diff_f_AE[k],diff_f_AE[k],PSD_AE[k],delta_f) for k in range(N_channels)])
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

response_kwargs = dict(TDItag="XYZ", tdi2=False)
wave_gen = BBHWaveformFD(amp_phase_kwargs=dict(run_phenomd=False), response_kwargs=response_kwargs) 

m1 = 1323277.47932  / (1 + 1.73941)
m2 =  612485.50602999  / (1 + 1.73941)
a1 = 0.747377 # spin 1
a2 =  0.8388   # spin 2
inc = np.pi/3 #inclination
dist_Gpc = 13.47098355897 # luminosity distance

beta = -0.30300442294174235  # ecliptic latitude
lam =   1.2925183861048521 # ecliptic longitude
# psi = np.pi/6 # polarization angle
psi = np.pi/3 # polarization angle

# t_ref = 2627744.9218792617
t_ref = 11526944.921879262
f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2 # phase at f_ref

M = (m1 + m2) 
q = m2 / m1  # m2 less than m1 

modes = [(2,2)]#, (2,1), (3,3), (3,2), (4,4), (4,3)]

params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]) 
N_params = len(params)

delta_t = 5.0
T_obs = 1*31*24*3600  # 2 months of data + padding on either side
delta_f = 1/T_obs 

time = np.arange(0, T_obs, delta_t)
# freq = xp.arange(1e-5,1e-1,delta_f)
freq = xp.arange(0.0,1e-1,delta_f)

kwargs = {"freq" : freq,
          "delta_f" : delta_f,
          "f_ref" : f_ref,
          "modes" : modes}

MBH_XY = MBH_f(wave_gen, time, *params, **kwargs)

##===========================Convert into time-domain ===================
MBH_XY_t = xp.asarray([(1/delta_t)*xp.fft.irfft(MBH_XY[k]) for k in range(N_channels)])
sim_t = xp.arange(len(MBH_XY_t[0])) * delta_t

# Full path to the data file
data_path = os.path.join(BH_directory, data)

# Read the HDF5 file
with h5py.File(data_path, 'r') as f:
    print("Keys in HDF5 file:")
    print(list(f.keys()))

    MBH_strain = f['time_series'][:]
    
    # Print structure of the file
    def print_structure(name, obj):
        print(name)
    f.visititems(print_structure)

import matplotlib.pyplot as plt 
time_spritz = MBH_strain[:,0]
time_spritz = time_spritz #time_spritz - time_spritz[0]
breakpoint()
MBH_X = MBH_strain[:,1]


plt.figure(figsize=(10, 6))
plt.plot(time_spritz, MBH_X, label = 'Spritz') 
plt.plot(sim_t, MBH_XY_t[0], label = 'BBHx')
plt.axvline(x = t_ref, c = 'red', linestyle = 'dashed', label = 't_ref')
plt.legend()
plt.xlim([2.6255e6,2.6285e6])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('MBH Waveform - Channel A (Time Domain)')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.clf()
# Frequency domain comparison
# Generate frequency bins and compute FFTs
fft_spritz = xp.fft.rfft(MBH_X)[1:]
freq_spritz = xp.fft.rfftfreq(len(MBH_X), d=time_spritz[1] - time_spritz[0])[1:]

fft_bbhx = xp.fft.rfft(MBH_XY_t[0])[1:]
freq_bbhx = xp.fft.rfftfreq(len(MBH_XY_t[0]), d=delta_t)[1:]

plt.figure(figsize=(10, 6))
plt.loglog(freq_spritz, xp.abs(fft_spritz), c = 'blue', label='Spritz', alpha=0.7)
plt.loglog(freq_bbhx, freq_bbhx * xp.abs(fft_bbhx), c = 'red', label='BBHx', alpha=0.7)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.title('MBH Waveform - Channel A (Frequency Domain)')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.show()


