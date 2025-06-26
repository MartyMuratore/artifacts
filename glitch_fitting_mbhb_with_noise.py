
## ----------- Pipeline for fitting for a glitch and an/or Astrophysical signals-------  ##

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
from eryn.state import BranchSupplemental
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

## wheater you want to generate data or not
generate_data= True

## which tipe of templeate you want to use in your likelihood 

only_mbhb=True ## only fitting for MBHB
only_glitch = False ## only fitting for glitches
mbh_glitch=False ## fitting for both

if only_mbhb==True: 
    branch_names = ["mbh"]
    ndims = {"mbh": 11}
    nleaves_max = {"mbh": 1}
    nleaves_min = {"mbh": 1}

if only_glitch==True:
    branch_names = ["glitch"]
    ndims = {"glitch": 3}
    nleaves_max = {"glitch": 1}
    nleaves_min = {"glitch": 1}

if mbh_glitch==True: 
    branch_names = ["glitch","mbh"]
    ndims = {"glitch": 3,"mbh": 11}
    nleaves_max = {"glitch": 1 , "mbh":1}
    nleaves_min = {"glitch": 1 ,  "mbh":1}

## chose the number of walkers and temperature to be used 
nwalkers = 30
ntemps = 10

# to sample the priors at high temperature
Tmax = np.inf

tempering_kwargs=dict(ntemps=ntemps,Tmax=Tmax)

## ---------- likelihood definition  ------- ##

def log_like_fn(x,bbhx_waveform_gen, bbh_transform_fn, data, psd, df,freqs,time,dt,subset = 1):
    if mbh_glitch==True: 
        glich_par = x[0]
        mbh_par = x[1]

        inds = np.arange(0, mbh_par.shape[0] + 1, subset)
        
        if inds[-1] < mbh_par.shape[0]:
            inds = np.concatenate([inds, np.array([mbh_par.shape[0]])])
        logl_all = []

        for i in range(len(inds) - 1):

            start = int(inds[i])
            end = int(inds[i + 1])
            shapelet_params =glich_par[start:end]
            bbhx_params =mbh_par[start:end]

            # computing the shapelet templeate 
            templateXYZ = combine_shapelet_link12_frequency_domain(freqs,shapelet_params,tdi_channel='first_gen',xp=xp).squeeze()
            A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2])
            fft_template = xp.asarray([A, E, T])

            # computing the MBHB templeate 
            bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=t_in[0]/one_year, t_obs_end=t_in[-1]/one_year,shift_t_limits=True)
            bbh_params_in = bbh_transform_fn.both_transforms(bbhx_params, return_transpose=True)
            AET_mbh = bbhx_waveform_gen(*bbh_params_in, **bbh_kwargs).squeeze()

            ## summing the templeate 
            fft_template += AET_mbh
        
            xp.get_default_memory_pool().free_all_blocks()
            
            ## when you want to plot the templeate and the data
            plot = False
            if plot == False:
                ## frequence plot
                plt.figure()
                fft_template = AET_mbh
                template_numpy_A =fft_template.get()[0]
                data_numpy_A = data.get()[0]       
                plt.loglog(freqs.get(),np.sqrt(2*np.abs(data_numpy_A)**2/dt/len(data_numpy_A)),'b-',label='data')
                plt.loglog(freqs.get(),np.sqrt(2*np.abs(template_numpy_A)**2/dt/len(template_numpy_A)),'k-',label='templeate')
                plt.loglog(freqs.get(),np.sqrt(psd.get()[0]),'y-',label='psd')
                plt.legend()
                plt.grid()
                plt.savefig("FFT_template_data_A.png")
                
                ## time plot
                n = len(data_numpy_A)
                time_axis = np.arange(0, n) / df
                time_domain_data_A = np.fft.ifft(data_numpy_A)
                time_domain_template_A = np.fft.ifft(template_numpy_A)
                plt.figure()    
                plt.plot(time_axis[61000:70000],time_domain_data_A[61000:70000],'b',label='data A time domain')
                plt.plot(time_axis[61000:70000],np.real(time_domain_template_A[61000:70000]),'r-',label='template A time domain')
                plt.legend()
                plt.grid()
                plt.savefig("time_template.png")
        
            ## computing the likelihood 

            logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data - fft_template)).real /psd, axis=0).sum())
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :ndim_glitch])
                breakpoint()  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)

    if only_mbhb==True: 
        mbh_par = x

        inds = np.arange(0, mbh_par.shape[0] + 1, subset)
        
        if inds[-1] < mbh_par.shape[0]:
            inds = np.concatenate([inds, np.array([mbh_par.shape[0]])])
        logl_all = []

        for i in range(len(inds) - 1):

            start = int(inds[i])
            end = int(inds[i + 1])
            bbhx_params =mbh_par[start:end]

            bbh_kwargs = dict(freqs=xp.asarray(freqs), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=t_in[0]/one_year, t_obs_end=t_in[-1]/one_year,shift_t_limits=True)
            bbh_params_in = bbh_transform_fn.both_transforms(bbhx_params, return_transpose=True)
            AET_mbh = bbhx_waveform_gen(*bbh_params_in, **bbh_kwargs).squeeze()
            fft_template = AET_mbh
        
            xp.get_default_memory_pool().free_all_blocks()

            plot = False
            if plot == True:
                ## frequence plot
                plt.figure()
                template_numpy_A =fft_template.get()[0]
                data_numpy_A = data.get()[0]       
                plt.plot(freqs.get(),np.absolute(data_numpy_A),label='data A')
                plt.plot(freqs.get(),np.absolute(template_numpy_A),label='template A')
                plt.legend()
                plt.grid()
                plt.savefig("FFT_template_data_A.png")

                ## time plot
                n = len(data_numpy_A)
                time_axis = np.arange(0, n) / df
                time_domain_data_A = np.fft.ifft(data_numpy_A)
                time_domain_template_A = np.fft.ifft(template_numpy_A)

                plt.figure()    
                plt.plot(time_axis[61000:70000],time_domain_data_A[61000:70000],'b',label='data A time domain')
                plt.plot(time_axis[61000:70000],np.real(time_domain_template_A[61000:70000]),'r-',label='template A time domain')
                plt.legend()
                plt.grid()
                plt.savefig("time_template.png")
        

            logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data - fft_template)).real /psd, axis=0).sum())
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :ndim_glitch])
                breakpoint()  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)

    if only_glitch==True: 
        glich_par = x

        inds = np.arange(0, glich_par.shape[0] + 1, subset)
        
        if inds[-1] < glich_par.shape[0]:
            inds = np.concatenate([inds, np.array([glich_par.shape[0]])])
        logl_all = []

        for i in range(len(inds) - 1):

            start = int(inds[i])
            end = int(inds[i + 1])
            shapelet_params =glich_par[start:end]

            templateXYZ = combine_shapelet_link12_frequency_domain(freqs,shapelet_params,tdi_channel='first_gen',xp=xp).squeeze()
            A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2])
            fft_template = xp.asarray([A, E, T])

            xp.get_default_memory_pool().free_all_blocks()
            
            plot = False
            if plot == True:
                ## frequence plot
                plt.figure()
                template_numpy_A =fft_template.get()[0]
                data_numpy_A = data.get()[0]       
                plt.plot(freqs.get(),np.absolute(data_numpy_A),label='data A')
                plt.plot(freqs.get(),np.absolute(template_numpy_A),label='template A')
                plt.legend()
                plt.grid()
                plt.savefig("FFT_template_data_A.png")

                ## time plot
                n = len(data_numpy_A)
                time_axis = np.arange(0, n) / df
                time_domain_data_A = np.fft.ifft(data_numpy_A)
                time_domain_template_A = np.fft.ifft(template_numpy_A)

                plt.figure()    
                plt.plot(time_axis[61000:70000],time_domain_data_A[61000:70000],'b',label='data A time domain')
                plt.plot(time_axis[61000:70000],np.real(time_domain_template_A[61000:70000]),'r-',label='template A time domain')
                plt.legend()
                plt.grid()
                plt.savefig("time_template.png")
        

            logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data - fft_template)).real /psd, axis=0).sum())
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :ndim_glitch])
                breakpoint()  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)
    
    return logl_out.get()

### ---------- Noise definition function  ------- ###

def noisepsd_AET(frq,Soms_d_in = (7.9e-12),Sa_a_in = (2.4e-15) , xp=None):

    import math

    lisaLT = 8.3  # LISA's armn in sec

    if xp is None:
        xp = np

    ## Acceleration noise
    ## In acceleration
    Sa_a = Sa_a_in**2 * (1.0 + (0.4e-3 / frq) ** 2) * (1.0 + (frq / 8e-3) ** 4)
    ## In displacement
    Sa_d = Sa_a * (2.0 * np.pi * frq) ** (-4.0)
    ## In relative frequency unit
    Sa_nu = Sa_d * (2.0 * np.pi * frq / C_SI) ** 2
    Spm = Sa_nu

    ### Optical Metrology System
    ## In displacement
    Soms_d = Soms_d_in**2 * (1.0 + (2.0e-3 / frq) ** 4)
    ## In relative frequency unit
    Soms_nu = Soms_d * (2.0 * np.pi * frq / C_SI) ** 2
    Sop = Soms_nu

    x = 2.0 * math.pi * lisaLT * frq

    Sa = (
        8.0
        * xp.sin(x) ** 2
        * (
            2.0 * Spm * (3.0 + 2.0 * xp.cos(x) + xp.cos(2 * x))
            + Sop * (2.0 + xp.cos(x))
        )
    ) 

    St = (16.0 * Sop * (1.0 - np.cos(x)) * np.sin(x) ** 2
        + 128.0 * Spm * np.sin(x) ** 2 * np.sin(0.5 * x) ** 4
    )

    return Sa, St

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

## ---- Glitch waveform in frequency domain ----- ###

dt = 0.25 # sampling time
Tobs = 1/12* YRSID_SI 
N = int(Tobs / dt)
Tobs = dt * N  # observation time of the mission in int. number
t_in = np.arange(N) * dt   # total time of the mission

freqs = np.fft.rfftfreq(len(t_in), dt)  # fs =1/dt
df = freqs[1] - freqs[0]  # 1 / (dt * len(t_in))

Xn , Yn ,Zn = combine_single_exp_glitch_link12_frequency_domain(freqs, np.asarray(glitch_params), T=8.3 , tdi_channel='first_gen', xp=None).squeeze()
Anfft,Enfft,Tnfft = AET(Xn, Yn, Zn)

## -------- Data glitch generation ------- ##
data_glitch_AET = xp.array([Anfft,Enfft,Tnfft])
    
## -------- Deciding the Frequency interval --- #####

fmin = 3e-5
fmax = 2.5e-2
frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them
freqs_cut = np.array(freqs[frequencymask])

### -- PSD noise -- ##

psd_AT = noisepsd_AET(freqs,Soms_d_in = (7.9e-12),Sa_a_in = (2.4e-15),  xp=None)
psd = np.array([psd_AT[0], psd_AT[0], psd_AT[1]])

## -----  Generate noise fft ------- ##

fft_noise_A = xp.asarray(get_sinthetic_noise(freqs,df,psd[0]))
fft_noise_E = xp.asarray(get_sinthetic_noise(freqs,df,psd[1]))
fft_noise_T = xp.asarray(get_sinthetic_noise(freqs,df,psd[2]))

## -----  Computing the synthetic PSD ------- ## 

psd_syn_A = get_sinthetic_psd(df,fft_noise_A )
psd_syn_E = get_sinthetic_psd(df,fft_noise_E )
psd_syn_T = get_sinthetic_psd(df,fft_noise_T )

## Check if the noise FFT is correctly generated ##
plt.figure()
plt.loglog(freqs, psd_syn_A.get(),'b',label='synthetic noise generation')
plt.loglog(freqs, psd_AT[0],'m-',label='PSD model')
plt.legend()
plt.grid()
plt.savefig("syntetic_movel_psd.png")
plt.close()

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

### ----- SNR of injected glitches ---###

SNRA_glitches = []
for i in range(len(glitch_params)):
    template_single_glitch = tdi_glitch_link12_frequency_domain(freqs, t0=glitch_params[i][0],  Deltav=glitch_params[i][1], tau=glitch_params[i][2],xp=None).squeeze() 
    A_single_glitch, E_single_glitch, T_single_glitch = AET(template_single_glitch[0], template_single_glitch[1], template_single_glitch[2])
    SNR_glitch  = np.sqrt(4*np.sum((np.abs(A_single_glitch[1:])**2)/psd[0][1:]*df +(np.abs(E_single_glitch[1:])**2)/psd[1][1:]*df + (np.abs(T_single_glitch[1:])**2)/psd[2][1:]*df)) # to check the snr
    SNRA_glitches.append(SNR_glitch)
print('snr_glitch:',SNRA_glitches)

### ----- SNR of MBHB ---###

SNR_mbh  = np.sqrt(4*np.sum((np.abs(data_mbh_AET.get()[0][1:])**2)/psd[0][1:]*df +(np.abs(data_mbh_AET.get()[1][1:])**2)/psd[1][1:]*df + (np.abs(data_mbh_AET.get()[2][1:])**2)/psd[2][1:]*df)) # to check the snr
print('SNR mbh:',SNR_mbh )

#### --------------- Putting together all the data in frequency domain -------------- #####

## MBHB frequency data 
fft_data_mbh = xp.asarray([data_mbh_AET[0] , data_mbh_AET[1],data_mbh_AET[2]])

## Glitch frequency data 
fft_data_glitch = xp.asarray([data_glitch_AET[0], data_glitch_AET[1],data_glitch_AET[2]])

## MBHB and noise
fft_data_mbh_noise = xp.asarray([fft_data_mbh[0] + fft_noise_A,fft_data_mbh[1] + fft_noise_E,fft_data_mbh[2] + fft_noise_T])
fft_data_mbh_noise = xp.asarray([fft_data_mbh[0][frequencymask] + fft_noise_A[frequencymask],fft_data_mbh[1][frequencymask] + fft_noise_E[frequencymask],fft_data_mbh[2][frequencymask] + fft_noise_T[frequencymask]])

## MBHB , glitch and noise
fft_data_all = xp.asarray([fft_data_mbh[0] + fft_data_glitch[0]+fft_noise_A,fft_data_mbh[1] + fft_data_glitch[1]+fft_noise_E,fft_data_mbh[2] + fft_data_glitch[2]+fft_noise_T])
fft_data_all_freqs_cut = xp.asarray([fft_data_all[0][frequencymask], fft_data_all[1][frequencymask],fft_data_all[2][frequencymask]])

## power spectral density computation

psd = np.array([psd[0][frequencymask], psd[1][frequencymask], psd[2][frequencymask]])
psd = xp.asarray(psd)
freqs_cut = xp.asarray(freqs_cut)

if __name__ == "__main__":
    
    ##  -----------    mbhb and glitch moves ----------- ##
    
    priors = {}
    if only_mbhb ==True:

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
                10: uniform_dist(t_ref-10,t_ref+10),
            }
        ) 

        ##  -----------    mbhb moves ----------- ##
    
        moves = [(SkyMove(which="both",gibbs_sampling_setup="mbh"), 0.04),
        (SkyMove(which="long",gibbs_sampling_setup="mbh"), 0.03),
        (SkyMove(which="lat",gibbs_sampling_setup="mbh"), 0.08),(StretchMove(gibbs_sampling_setup="mbh"),0.85)]  # the MCMC uses arbitrarly the weight

        ##  -----------   coordinats  if only mbhb----------- ##

        coords = {"mbh": np.zeros((ntemps, nwalkers, nleaves_max["mbh"], ndims["mbh"]))}

        # make sure to start near the proper setup
        inds ={
                "mbh": np.ones((ntemps, nwalkers, nleaves_max["mbh"]), dtype=bool)
            }

    
    if  mbh_glitch== True:
        gibbs = []
        nfriends = nwalkers
        for i in range(nleaves_max["glitch"]):
            tmp = np.zeros((nleaves_max["glitch"], ndims["glitch"]), dtype=bool)
            tmp[i] = True

            gibbs.append(("glitch", tmp))
    
        gs = group_stretch(nfriends=nfriends,gibbs_sampling_setup=gibbs,  n_iter_update=100) #
        scgm = SelectedCovarianceGroupMove(nfriends=1, gibbs_sampling_setup=gibbs,n_iter_update=100)
        
        ### update in case f different glitches
        moves = [ (gs, 0.5), (scgm, 0.5),(SkyMove(which="both",gibbs_sampling_setup="mbh"), 0.04),
        (SkyMove(which="long",gibbs_sampling_setup="mbh"), 0.03),
        (SkyMove(which="lat",gibbs_sampling_setup="mbh"), 0.08),(StretchMove(gibbs_sampling_setup="mbh"),0.85)]  # the MCMC uses arbitrarly the weight
  
 
    ##  -----------   priors  ----------- ##

        priors_in = {
            0: uniform_dist( t_ref-1e3,t_ref+1e3),  
            1: uniform_dist(-35,-20), #prior_distribution_dv, # i need to check this
            2: uniform_dist(1, 1e5)} # prior_distribution_tau}#prior_distribution_tau 

        priors["glitch"] = ProbDistContainer(priors_in)
    

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
                10: uniform_dist(t_ref-10,t_ref+10),
            }
        ) 
  
        ##  -----------   coordinats  ----------- ##
    
        coords = {
                "glitch": priors["glitch"].rvs(size=(ntemps, nwalkers,nleaves_max["glitch"])),
                "mbh": np.zeros((ntemps, nwalkers, nleaves_max["mbh"], ndims["mbh"]))}

        # make sure to start near the proper setup
        inds = {
                "glitch": np.ones((ntemps, nwalkers, nleaves_max["glitch"]), dtype=bool),
                "mbh": np.ones((ntemps, nwalkers, nleaves_max["mbh"]), dtype=bool)
            }

    ##### ------ initiating the MBHB --------- ###


    def log_like_mbh(x,bbhx_waveform_gen,  bbh_transform_fn, data, psd, df,freqs,time, subset = 1):

        mbh_par = x

        inds = np.arange(0, mbh_par.shape[0] + 1, subset)
        
        if inds[-1] < mbh_par.shape[0]:
            inds = np.concatenate([inds, np.array([mbh_par.shape[0]])])
        logl_all = []

        for i in range(len(inds) - 1):

            start = int(inds[i])
            end = int(inds[i + 1])
            bbhx_params =mbh_par[start:end]
            bbh_kwargs = dict(freqs=xp.asarray(freqs_cut), direct=False, fill=True, squeeze=True, length=1024, t_obs_start=time[0]/one_year, t_obs_end=time[-1]/one_year,shift_t_limits=True)
            bbh_params_in = bbh_transform_fn.both_transforms(bbhx_params, return_transpose=True)
            AET_mbh = bbhx_waveform_gen(*bbh_params_in, **bbh_kwargs).squeeze()
            fft_template = AET_mbh
        
            xp.get_default_memory_pool().free_all_blocks()

            logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data - fft_template)).real /psd, axis=0).sum())
            logl = logl[np.newaxis]
            if xp.any(xp.isnan(logl)):
                print("nans:", tmp[fix, :ndim_glitch])
                breakpoint()  
            logl_all.append(logl)
            
        logl_out = np.concatenate(logl_all)
    
        return logl_out.get()

    factor = 1e-9
    cov = np.ones(ndims['mbh']) * 1e-3
    cov[3] = 1e-7
    cov[-1] = 1e-7

    start_like = np.zeros((nwalkers * ntemps))
        
    iter_check = 0
    max_iter = 1000
    while np.std(start_like) < 10:
        
        logp = np.full_like(start_like, -np.inf)
        tmp = np.zeros((ntemps * nwalkers,ndims['mbh'] ))
        fix = np.ones((ntemps * nwalkers), dtype=bool)
        while np.any(fix):
                        
            tmp[fix, :] = (mbh_injection_params[None, :] * (1. + factor * cov * np.random.randn(nwalkers * ntemps, ndims['mbh'])))[fix]

            # Apply modulo operations to the MBH parameters only
            tmp[:, 5] = tmp[:, 5] % (2 * np.pi)
            tmp[:, 7] = tmp[:, 7] % (2 * np.pi)
            tmp[:, 9] = tmp[:, 9] % (1 * np.pi)

            logp = priors["mbh"].logpdf(tmp)

            fix = np.isinf(logp)
            if np.all(fix):
                breakpoint()

        x = tmp

        start_like =log_like_mbh(x, wave_gen, transform_fn, xp.asarray([data_mbh_AET[0][frequencymask] , data_mbh_AET[1][frequencymask],data_mbh_AET[2][frequencymask]]), psd, df,freqs_cut,t_in)
        
        iter_check += 1
        factor *= 1.5

        print(np.std(start_like))

        if iter_check > max_iter:
            raise ValueError("Unable to find starting parameters.")

    coords["mbh"] =  tmp.reshape(ntemps, nwalkers,nleaves_max["mbh"], ndims["mbh"])

    

    fp = "hightSNRmbhb_noise_glitchSNR1544_merger.h5"
      
    def update_fn(i, res, samp):
        max_it_update=400000
        mem =500

 
        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()))
        print("min last loglike",np.min(samp.get_log_like()))
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
           # print("rj \n",samp.rj_acceptance_fraction[0] )
            print("swap \n",samp.swap_acceptance_fraction)

        
        likelihood = samp.get_log_like()[-mem:, 0,:]
        
        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.xlabel('iter')
        plt.ylabel('Likelihood (single glitch)')
        plt.title('likelihood evolutions for all the walkers at 0 temperature with glitch')
        plt.grid(True)
        plt.show()
        plt.savefig("likelihood_freq_domain_mbh.png", dpi=300)
        plt.close()
      
        
        if (samp.iteration<max_it_update):
            if only_mbhb == True:
                import matplotlib as mpl
                mpl.rcParams['text.usetex'] = False  # Add this before any plotting
                item_samp_mbh = samp.get_chain()["mbh"][-mem:,0][~np.isnan(samp.get_chain()["mbh"][-mem:,0])].reshape(-1,ndims['mbh'])
                parameter_labels = ['$lnM$', '$q$', '$a1$', '$a2$', '$dL$', '$phi$', '$theta$', '$lambda$', '$beta$', '$psi$', '$t$']
                fig = corner.corner(
                            item_samp_mbh,
                            truths=mbh_injection_params,
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

                plt.savefig("mbh_corner_plot.png", dpi=300)
                plt.close()
           
            if mbh_glitch == True:
                item_samp_glitch_no_nans= samp.get_chain()['glitch'][-mem:,0][~np.isnan(samp.get_chain()['glitch'][-mem:,0])].reshape(-1,3)

                gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-15)

                gmm_glitch.fit(item_samp_glitch_no_nans)


                for move in samp.moves:
                    if hasattr(move, "name") and move.name == "selected covariance":
                        move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)

                
                c = ChainConsumer()
                parameter_labels = ['$\\tau$', '$ln(dv)$', '$\\beta$']
    
                c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels, name='Glitches', color='#6495ed')
                
                c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
                
                for ii in range(len(glitch_params)):
                    c.add_marker([glitch_params[ii][0], np.log(np.exp(glitch_params[ii][1])/(2*glitch_params[ii][2])), glitch_params[ii][2]], \
                    parameters=parameter_labels, marker_style="x", \
                    marker_size=100, color='#DC143C')
                
                fig = c.plotter.plot(figsize=(8,8), legend=True)
                plt.savefig("glitches_corner_plot.png", dpi=300)
                plt.close()
            
                    
        if (samp.iteration==max_it_update):
            mem = int(samp.iteration*0.9)
          
            c = ChainConsumer()
            # Define the parameter names with LaTeX labels
            parameter_labels = [r'$\ln M$', r'$q$', r'$a_1$', r'$a_2$', r'$d_L$', r'$\phi$', r'$\cos(\theta)$', r'$\lambda$', r'$\beta$', r'$\psi$', r'$t$']
            item_samp_mbh = samp.get_chain()["mbh"][-mem:,0][~np.isnan(samp.get_chain()["mbh"][-mem:,0])].reshape(-1,ndims['mbh'])
            c.add_chain(item_samp_mbh, parameters=parameter_labels , name='mbh', color='#6495ed')

            c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
            
            fig = c.plotter.plot(figsize=(8,8), legend=True)
            plt.savefig("mbh_corner_plot.png", dpi=300)
            plt.close()


            item_samp_glitch_no_nans= samp.get_chain()[el][-mem:,0][~np.isnan(samp.get_chain()[el][-mem:,0])].reshape(-1,3)

            gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-10)

            gmm_glitch.fit(item_samp_glitch_no_nans)

            #### compute glitch covariance matrix ###

            for move in samp.moves:
                if hasattr(move, "name") and move.name == "selected covariance":
                    move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)
                    ##########
                

            #   samp.moves[1].all_proposal[el].scale = 2.38**2 / ndims * (np.cov(item_samp,rowvar=False) )
            parameter_labels = ['$t0$', '$ln(DV)$', '$\\tau$']
            c = ChainConsumer()

            c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
            
            c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)

            for ii in range(len(glitch_params)):
                c.add_marker([glitch_params[ii][0], glitch_params[ii][1], glitch_params[ii][2]], \
                parameters=parameter_labels, marker_style="x", \
                marker_size=100, color='#DC143C')

            fig = c.plotter.plot(figsize=(8,8), legend=True)
            plt.savefig("glitches_corner_plot_end.png", dpi=300)

        return False



    if True:

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
            args=[wave_gen, transform_fn,fft_data_all_freqs_cut , psd,df,freqs_cut,t_in,dt],
            tempering_kwargs=tempering_kwargs,
            moves=moves,
            provide_groups=False,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            branch_names=branch_names,
            update_iterations=1,
            update_fn=update_fn,
            nbranches=1, # to change in case we use two branches -->mbh_glitch=true
            vectorize=True,
            backend= HDFBackend(fp))
            
        nsteps =50000
        thin_by=1
        burn=0
            
    
        
        print('start')

        log_prior = ensemble.compute_log_prior(coords, inds=inds)

        log_like = ensemble.compute_log_like(coords, inds=inds, logp=log_prior)[0]
        if mbh_glitch==True:
            ## setup branch supplemental to carry group stretch information
            closest_inds = -np.ones((ntemps, nwalkers, nleaves_max["glitch"], nfriends), dtype=int)
            closest_inds_cov = -np.ones((ntemps, nwalkers, nleaves_max["glitch"]), dtype=int)
        
            branch_supps = {
                "glitch": BranchSupplemental(
                    {"inds_closest": closest_inds, "inds_closest_cov": closest_inds_cov}, base_shape=(ntemps, nwalkers, nleaves_max["glitch"])),
                    "mbh": None}
        if only_mbhb==True:
            branch_supps = { "mbh": None}

        
        start_state = State(coords, inds=inds , log_like=log_like, log_prior=log_prior,branch_supplemental=branch_supps)

        out = ensemble.run_mcmc(start_state, nsteps, burn=burn, progress=True, thin_by=thin_by)





