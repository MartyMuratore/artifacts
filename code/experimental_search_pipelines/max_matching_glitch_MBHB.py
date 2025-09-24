### -------------- Compute inner product for the maximum matching ----------- ####

# General tools 

import os
import matplotlib.pyplot as plt
import corner
import cupy as cp
import numpy as np
from chainconsumer import ChainConsumer
import scipy.stats as stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Import module for Eryn usage

from eryn.ensemble import EnsembleSampler
from eryn.state import State, BranchSupplemental
from eryn.prior import ProbDistContainer, uniform_dist,  log_uniform
from lisatools.Gaussian_prior import gaussian_dist
from eryn.utils import TransformContainer
from eryn.backends import HDFBackend
from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.constants import *
from bbhx.utils.transform import *

# import moves

from eryn.moves import GaussianMove, StretchMove, GroupStretchMove , GroupMove,   DistributionGenerate
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.glitch_shapelet_analytical_waveform import combine_shapelet_link12_frequency_domain, tdi_shapelet_link12_frequency_domain,combine_single_exp_glitch_link12,tdi_glitch_link12_frequency_domain,combine_single_exp_glitch_link12_frequency_domain
from lisatools.group_stretch_proposal import MeanGaussianGroupMove as group_stretch
from lisatools.group_stretch_proposal import SelectedCovarianceGroupMove

# generate noise 
from synthetic_noise_generator import get_sinthetic_noise, get_sinthetic_psd

# import MBHB weveform
from bbhx.waveformbuild import BBHWaveformFD
from bbhx.waveforms.phenomhm import PhenomHMAmpPhase
from bbhx.response.fastfdresponse import LISATDIResponse

# set random seed
np.random.seed(10)


## set the GPU to use
try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(6)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using GPU or not
use_gpu = True

if use_gpu is False:
    xp = np

    ## ------------ Waveform MBHB  --------- ##

f_ref = 0.0 # let phenom codes set f_ref -> fmax = max(f^2A(f))
phi_ref = 1.2 # phase at f_ref
m1 = 4.5*10**7 
m2 = 1.5*10**7  
M = (m1 + m2)
q = m2 / m1  # m2 less than m1 
a1 = 0.3
a2 =  0.4  # a1 >a2
dist = 47.6
inc = 0.6
beta = 0.3  # ecliptic latitude
lam =  2.0 # ecliptic longitude
psi = 1.7 # polarization angle
t_ref =  30*60*60 #11526944.921879262


# wave generating class
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=dict(run_phenomd=False,initial_t_val = t_in[0]),
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
mbh_injection_params[8] = np.sin(mbh_injection_params[8])  # Takes the sine of the polarization angle.


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
periodic = {
    "mbh": {5: 2 * np.pi, 7: np.pi, 8: np.pi}
}

########  --------- DATA MBHB  --------- ### 

one_year = 86400*365.26
bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=t_in[0]/one_year, t_obs_end=t_in[-1]/one_year,shift_t_limits=True)
injection_in = transform_fn.both_transforms(mbh_injection_params[None, :], return_transpose=True)
data_mbh_AET = wave_gen(*injection_in, **bbh_kwargs)[0]

# Define function to compute match
def compute_match(H1, H2, psd, df):
    """Computes the normalized match between two signals H1 and H2."""
    inner_product = lambda A, B: 4 * np.real(np.sum(A * np.conj(B) / psd) * df)
    norm_h1 = np.sqrt(inner_product(H1, H1))
    norm_h2 = np.sqrt(inner_product(H2, H2))
    return inner_product(H1, H2) / (norm_h1 * norm_h2)

###  -------- LPF glitches from the catalogue  ------ ###

file_path = 'glitch_params_all_PRDLPF.h5'
import h5py
with h5py.File(file_path, 'r') as hdf:
        dv = hdf['dv'][:]
        t0 = hdf['t0'][:]
        tau1 = hdf['tau1'][:]
        tau2 = hdf['tau2'][:]
pathfinder_glitch_dist=np.array([t0,np.log(np.abs(dv)),tau1,tau2]) 

##### --------------- Data generation -------------------- ######

## to create a list with all the injected glitches 

glitch_from_lpf = []
num_of_glitch = []
numbers = [40,2,247,0]  #40 is 21 ;  #2 is 72 ; and #247 is 1544, #0 is 20

for random_number in numbers:

    glitch = pathfinder_glitch_dist[:, random_number]
    glitch_from_lpf.append(glitch)

## this part is for choosing the injection time since the LPF injection time is different from LISA orbit
glitch_params = np.array(glitch_from_lpf)[:,:3]
t_inj = 108423.84769539078
glitch_params[0,0] = t_inj


# Glitch SNR values for legend
snr_values = {40: 21, 2: 72,247: 1154}
colors = {40: 'red', 2: 'blue', 247: 'green', 0: 'magenta'}

glitch_from_lpf = []
numbers = [40,2]


for random_number in numbers:
    glitch = pathfinder_glitch_dist[:, random_number]
    glitch_from_lpf.append(glitch)

glitch_params_list = np.array(glitch_from_lpf)[:, :3]
t_ref_values = np.linspace(t_ref - 1500, t_ref + 1500, 5)  # Example time scan range

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


# Glitch SNR values for legend
snr_values = {41: 21, 3: 72, 248: 1544}
colors = {40: 'red',2: 'blue', 247: 'green'}

glitch_from_lpf = []
numbers = [40, 2]


for random_number in numbers:
    glitch = pathfinder_glitch_dist[:, random_number]
    glitch_from_lpf.append(glitch)

glitch_params_list = np.array(glitch_from_lpf)[:, :3]
t_ref_values = np.linspace(t_ref - 2000, t_ref + 2000, 500)  # Example time scan range



plt.figure(figsize=set_figsize('single', ratio =0.8))

for i, (random_number, glitch_params) in enumerate(zip(numbers, glitch_params_list)):
    best_match = -np.inf
    best_t_ref = None
    matches = []


    H1 = data_mbh_AET.get()[0][1:]  # Reference data

    for t_inj in t_ref_values:
        # Update t_ref in glitch parameters
        glitch_params[0] = t_inj  

        # Compute glitch waveform in frequency domain
        Xn, Yn, Zn = combine_single_exp_glitch_link12_frequency_domain(
            freqs, np.asarray([glitch_params]), T=8.3, tdi_channel='first_gen', xp=None
        ).squeeze()
        Anfft, Enfft, Tnfft = AET(Xn, Yn, Zn)
        data_glitch_AET = np.array([Anfft, Enfft, Tnfft])

        H2 = data_glitch_AET[0][1:]

        # Compute match
        match = compute_match(H1, H2, psd[0][1:], df)
        matches.append(match)

        if match > best_match:
            best_match = match
            best_t_ref = t_inj


        # Plot results for this glitch

    plt.plot(
        (t_ref_values- t_ref)/(60), matches, linestyle='-', color=colors[random_number],
        label=f'Glitch $\#${random_number+1}'
    )
    plt.axvline((best_t_ref-t_ref)/(60), color=colors[random_number], linestyle='--', label = f'Best Overlap Glitch $\#${random_number + 1}')
    print('best t_inj',best_t_ref)


plt.xlabel("$t_{inj}(Glitch)$ - $t_{ref}(MBHB)$ [minutes]")
plt.ylabel("Overlap")
plt.legend(loc ='best')
plt.savefig("max_matching_all_glitches.png")
plt.show()

## Plots comparing glitch maximum matching with the MBHB

plt.figure()
plt.loglog(freqs, np.sqrt(2*np.abs(data_glitch_AET.get()[0])**2/dt/len(data_glitch_AET.get()[0])), 'g--', label="Glitches signals only", alpha=1, linewidth=1)
plt.loglog(freqs,  np.sqrt(2*np.abs(data_mbh_AET.get()[0])**2/dt/len(data_mbh_AET.get()[0])), 'b', label="MBHB signal only", alpha=1, linewidth=1)
plt.loglog(freqs, np.sqrt(psd[0]), 'k--', label="PSD noise model", alpha=1, linewidth=1)
#plt.plot(t_in/(60*60),time_domain_data_mbh_E,'r',label='First generation-TDI $E$')
#plt.plot(t_in/(60*60),time_domain_data_mbh_T,'k--',label='First generation-TDI $T$')
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.ylim([-0.5*1e-22,1e-22])
plt.ylabel("Strain", fontsize=15)
plt.xlabel("Frequency [Hz]", fontsize=15)
#plt.xlim([729.1, 730.05])
#plt.xticks([729.1,729.4,729.7,730.05], fontsize=15)
plt.legend(loc = 'lower left', fontsize=11)
plt.savefig("freq_template_lowmbhb_withSNR72glitch.png", dpi=300, bbox_inches='tight')
plt.close()



plt.figure()
time_domain_data_mbh_A = np.fft.irfft(data_mbh_AET.get()[0],len(t_in))
time_domain_data_mbh_E = np.fft.irfft(data_mbh_AET.get()[1],len(t_in))
time_domain_data_mbh_T = np.fft.irfft(data_mbh_AET.get()[2],len(t_in))
time_domain_data_glitch_A = np.fft.irfft(data_glitch_AET.get()[0,1:],len(t_in[1:]))
plt.plot(t_in[1:]/(60*60),time_domain_data_glitch_A,'g',label='Glitch first generation-TDI $A$ SNR21')
plt.plot(t_in/(60*60),time_domain_data_mbh_A,'b',label='First generation-TDI $A$')
plt.plot(t_in/(60*60),time_domain_data_mbh_E,'r',label='First generation-TDI $E$')
plt.plot(t_in/(60*60),time_domain_data_mbh_T,'k--',label='First generation-TDI $T$')
plt.grid()
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
#plt.ylim([-0.5*1e-22,0.5*1e-22])
plt.ylabel("Strain", fontsize=15)
plt.xlabel("Time [h]", fontsize=15)
plt.xlim([24, 31])
plt.xticks([24,26,28,31], fontsize=15)
plt.legend(loc = 'lower left', fontsize=11)
plt.savefig("time_template_lowmbhb_withSNR21glitch.png", dpi=300, bbox_inches='tight')
plt.close()