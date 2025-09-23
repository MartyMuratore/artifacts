## ----------- Pipeline for fitting for multiple glitches without a Astrophysical signals with RJ -------  ##

# import dependences 
import os
import matplotlib.pyplot as plt
import corner
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from chainconsumer import ChainConsumer
import scipy.stats as stats
from scipy.stats import multivariate_normal
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Import Eryn sampler, priors and moves 

from eryn.ensemble import EnsembleSampler
from eryn.state import BranchSupplemental
from eryn.state import State, BranchSupplemental
from eryn.prior import ProbDistContainer, uniform_dist,  log_uniform
from lisatools.Gaussian_prior import gaussian_dist
from eryn.utils import TransformContainer
from eryn.backends import HDFBackend
from eryn.moves import GaussianMove, StretchMove, GroupStretchMove , GroupMove,   DistributionGenerate
from lisatools.sampling.likelihood import Likelihood
from lisatools.sampling.moves.skymodehop import SkyMove
from lisatools.glitch_shapelet_analytical_waveform import combine_shapelet_link12_frequency_domain, tdi_shapelet_link12_frequency_domain,combine_single_exp_glitch_link12,tdi_glitch_link12_frequency_domain,combine_single_exp_glitch_link12_frequency_domain
from lisatools.group_stretch_proposal import MeanGaussianGroupMove as group_stretch
from lisatools.group_stretch_proposal import SelectedCovarianceGroupMove
from lisatools.utils.utility import AET
from lisatools.sensitivity import get_sensitivity
from bbhx.utils.constants import *
from bbhx.utils.transform import *
from synthetic_noise_generator import get_sinthetic_noise, get_sinthetic_psd

# set the GPU to use
try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(4)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# whether you are using GPU or not
use_gpu = True

if use_gpu is False:
    xp = np


# data gliches and noise
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

# set random seed
np.random.seed(10)

# Defining the global set up for the sampler (branches, leaves, walkers and temperatures)

branch_names = ["glitch","noise"]
ndims = {"glitch": 3, "noise": 2}
nleaves_max =  {'glitch':10,"noise": 1} 
nleaves_min =   {'glitch':0,"noise": 1} 
nwalkers = 70
ntemps = 10

tempering_kwargs=dict(ntemps=ntemps,Tmax=np.inf)

# definition of the log_likelihood

def log_like_fn(x_param, groups, data, df, freqs):   
 
    glitch_params_all, noise_params_all =  x_param

    group_glitch, group_noise  = groups

    ngroups = int(group_noise.max()) + 1

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

        noise_params = noise_params_all[group]
        
        # noise estimation 
        psd = noisepsd_AET(freqs,Soms_d_in =  noise_params[0],Sa_a_in =  noise_params[1])

        tot_psd =  xp.asarray([psd[0],psd[0], psd[1]]) ## to account for the filter
     
        # glitch estimation 
        templateXYZ = combine_shapelet_link12_frequency_domain(freqs,shapelet_params,tdi_channel='first_gen',xp=xp).squeeze()
        
        A, E, T = AET(templateXYZ[0], templateXYZ[1], templateXYZ[2]) ## to account for the filter

        fft_template = xp.asarray([A, E, T])
     
        xp.get_default_memory_pool().free_all_blocks()
        
        # evaluetate the log likelihood
 
        logl = -1/2 * (4*df* xp.sum((xp.conj(data-fft_template) *(data-fft_template )).real /(tot_psd ), axis=0).sum() )
        logl += -  xp.sum(xp.log(tot_psd), axis=0).sum()
        logl = logl[np.newaxis]

        # check if we have nan in the log_likelihood
        if xp.any(xp.isnan(logl)):
                print("nans:", coords['glitch']) 
                breakpoint()  

        # I need to create an array with all logl that i have computed for each iteration    
        logl_all.append(logl)
    
        # I am concatenating over the logl_all that I am computing at each iteration to have a vector of the log_likelihood computed
        logl_out = np.concatenate(logl_all)
    return  logl_out.get()


# This is the file containing the LPF glitches
file_path = 'glitch_params_all_PRDLPF.h5'
import h5py
with h5py.File(file_path, 'r') as hdf:
        dv = hdf['dv'][:]
        t0 = hdf['t0'][:]
        tau1 = hdf['tau1'][:]
        tau2 = hdf['tau2'][:]
pathfinder_glitch_dist=np.array([t0,np.log(np.abs(dv)),tau1,tau2]) 

#this loop is for defining which glitch to use, it is possible also to pick them randomly

glitch_from_lpf = []
num_of_glitch = []
numbers = [2,117,89,71,178,247,40]  #4 is 7.4 ; 40 is 21; 13 is 14.6; # 204 is 19.9, # 74 is 27 , # 98 is 31

for random_number in numbers:
    glitch = pathfinder_glitch_dist[:, random_number]
    glitch_from_lpf.append(glitch)

glitch_params = np.array(glitch_from_lpf)[:,:3]

# ------------ data generation  ----------- #

# time definition to generate the time series

dt = 0.25 # sampling time
Tobs = 1/12* YRSID_SI 
N = int(Tobs / dt)
Tobs = dt * N  # observation time of the mission in int. number
t_in = np.arange(N) * dt   # total time of the mission

# here we distribute the time such that there are randomly distributed
glitch_params[0,0] = t_in[1190000]
glitch_params[1,0] = t_in[3000000]
glitch_params[2,0] = t_in[5000000]
glitch_params[3,0] = t_in[7000000]
glitch_params[4,0] = t_in[8800000]
glitch_params[5,0] = t_in[9000000]
glitch_params[6,0] = t_in[10000000]

print(glitch_params)

# ---------- frequency data ----------- #

freqs = np.fft.rfftfreq(len(t_in), dt)  # fs =1/dt

Xn , Yn ,Zn = combine_single_exp_glitch_link12_frequency_domain(freqs, np.asarray(glitch_params), T=8.3 , tdi_channel='first_gen', xp=None)

data_A_fft,data_E_fft,data_T_fft = AET(Xn, Yn, Zn)

# reducing the frequency axis to cut the zeros of the TDI 

fmin = 1e-4
fmax = 2e-2
frequencymask = (freqs > fmin) & (freqs < fmax) 
freqs_cut = np.array(freqs[frequencymask])
df = freqs[2] - freqs[1]  # 1 / (dt * len(t_in))

# defining the noise PSD
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

# generating the Fourier transform from the noise PSD

psd_AT = noisepsd_AET(freqs,Soms_d_in = (7.9e-12),Sa_a_in = (2.4e-15),  xp=None)
psd = np.array([psd_AT[0], psd_AT[0], psd_AT[1]])

### generate noise fft

fft_noise_A = get_sinthetic_noise(freqs,df,psd[0])
fft_noise_E = get_sinthetic_noise(freqs,df,psd[1])
fft_noise_T = get_sinthetic_noise(freqs,df,psd[2])

fft_data = xp.asarray([data_A_fft +fft_noise_A , data_E_fft + fft_noise_E, data_T_fft+ fft_noise_T]).squeeze()


# ------- plots of the synthetic noise -------- #
plt.figure(figsize=set_figsize('single', ratio =1))
plt.loglog(freqs ,  np.sqrt(get_sinthetic_psd(df,fft_data.get()[0])) ,'r',label='Noise and glitches data TDI $A$',alpha=0.8)
plt.loglog(freqs , np.sqrt(get_sinthetic_psd(df,fft_data.get()[1])),'y.-',label='Noise and glitches data TDI $E$',alpha=0.8)
plt.loglog(freqs ,   np.sqrt(get_sinthetic_psd(df,fft_data.get()[2])) ,'cyan',label='Noise and glitches data TDI $T$',alpha=0.8)
plt.loglog(freqs ,  np.sqrt(psd_AT[0] ) ,'k--',label='Model TDI $A/E$',alpha = 0.8)
plt.loglog(freqs,  np.sqrt(psd_AT[1])  ,'k:',label='Model TDI $T$',alpha = 0.8)
plt.ylim([1e-30,1e-18])
plt.grid(True)
plt.legend(loc="lower center", bbox_to_anchor=(0.5, 0.02))
plt.xlabel("Frequency [$Hz$]")  # Increase the font size of x-axis label
plt.ylabel("Amplitude spectral density [$Hz^{-1/2}$]")  # Increase the font size of y-axis label
plt.savefig("psd_tdi_glitches_freq_domain.png")
plt.close()

# compute SNR of the glitch signal considered

SNR_glitch =[]

for i in range(len(glitch_params)):

    template_single_glitch = tdi_glitch_link12_frequency_domain(freqs_cut, t0=glitch_params[i][0],  Deltav=glitch_params[i][1], tau=glitch_params[i][2],xp=None).squeeze() 

    # I consider here ortogonal channels A,E,T

    A_single_glitch, E_single_glitch, T_single_glitch = AET(template_single_glitch[0], template_single_glitch[1], template_single_glitch[2])

    SNR_glitches  = np.sqrt(4*np.sum((np.abs(A_single_glitch))**2/psd[0][frequencymask]*df +(np.abs(E_single_glitch))**2/psd[1][frequencymask]*df +(np.abs(T_single_glitch))**2/psd[2][frequencymask]*df)) # to check the snr

    SNR_glitch.append(SNR_glitches)
print('SNR A glitch:',SNR_glitch)

## compute the SNR of all the glitch signal in the LPF catalogue

SNR_all_true =False

if SNR_all_true==True:

    SNRA_glitch_dist = []

    for i in range(pathfinder_glitch_dist[:3,:].shape[1]):

        template_single_glitch = tdi_glitch_link12_frequency_domain(freqs, t0=pathfinder_glitch_dist[:3,:][:,i][0],  Deltav=pathfinder_glitch_dist[:3,:][:,i][1], tau=pathfinder_glitch_dist[:3,:][:,i][2],xp=None).squeeze() 

        A, E, T = AET(template_single_glitch[0], template_single_glitch[1], template_single_glitch[2])

        # computing the fft of the TDI 
        data_A = A[1:]
        data_E = E[1:]
        data_T = T[1:]

        psd_a = psd[0][1:]
        psd_e = psd[1][1:]
        psd_t = psd[2][1:]

        SNRA  = np.sqrt(4*np.sum((np.abs(data_A))**2/psd_a*df +(np.abs(data_E))**2/psd_e*df +(np.abs(data_T))**2/psd_t*df)) # to check the snr

        SNRA_glitch_dist.append(SNRA)

    pick_prior_snr25 = np.where((np.array(SNRA_glitch_dist) > 23) & (np.array(SNRA_glitch_dist) < 30))[0]
    pick_prior_snr15 = np.where(np.array(SNRA_glitch_dist) <15)[0]
    pick_prior_snr30 = np.where((np.array(SNRA_glitch_dist) > 15) & (np.array(SNRA_glitch_dist) < 30))[0]
    pick_prior_snr50 = np.where((np.array(SNRA_glitch_dist) > 30) & (np.array(SNRA_glitch_dist) < 50))[0]
    pick_prior_snr70 = np.where((np.array(SNRA_glitch_dist) > 50) & (np.array(SNRA_glitch_dist) < 70))[0]
    pick_prior_snr100 = np.where((np.array(SNRA_glitch_dist) > 70) & (np.array(SNRA_glitch_dist) < 100))[0]
    pick_prior_snr200 = np.where((np.array(SNRA_glitch_dist) > 100) & (np.array(SNRA_glitch_dist) < 200))[0]
    pick_prior_snr500 = np.where((np.array(SNRA_glitch_dist) > 200) & (np.array(SNRA_glitch_dist) < 500))[0]
    pick_prior_snr1000 = np.where((np.array(SNRA_glitch_dist) > 500) & (np.array(SNRA_glitch_dist) < 1000))[0]
    pick_prior_snr2000 = np.where(np.array(SNRA_glitch_dist) > 1000 )[0]



# defining the gpu array
psd = xp.asarray(psd)
freqs_cut = xp.asarray(freqs_cut)

# frequency cutted in frequency

fft_data = xp.asarray([fft_data[0][frequencymask],fft_data[1][frequencymask],fft_data[2][frequencymask]])

    
if __name__ == "__main__":

    # ------ prior definition -------- #

    priors = {}
    
    priors_in = {
        0: uniform_dist(t_in[0],t_in[-1]),  
        1: uniform_dist(-35,-20),
        2: uniform_dist(1, 1e5)} 

    priors["glitch"] = ProbDistContainer(priors_in)

    priors_noise = {
    0: uniform_dist((7e-12),(8e-12)),
    1: uniform_dist((2e-15),(3e-15)),}

    priors['noise'] = ProbDistContainer(priors_noise) 

    # ---------  moves definitions for glitches  ---------- #

    nfriends = nwalkers

    gibbs = []
    for i in range(nleaves_max["glitch"]):
        tmp = np.zeros((nleaves_max["glitch"], ndims["glitch"]), dtype=bool)
        tmp[i] = True

        gibbs.append(("glitch", tmp))
   
    gs = group_stretch(nfriends=nfriends,gibbs_sampling_setup=gibbs,  n_iter_update=100) #
    scgm = SelectedCovarianceGroupMove(nfriends=1, gibbs_sampling_setup=gibbs,n_iter_update=100)

    # ------ moves definitions for noise ---------- #
        
    factor = 1e-15
    cov = { 
    "noise": np.diag(np.ones(ndims['noise'])) * factor }

    # total moves
 
    moves = [ (gs, 0.5), (scgm, 0.5),(StretchMove(gibbs_sampling_setup ="noise"),0.85),(GaussianMove(cov,gibbs_sampling_setup ="noise"),0.15)] 
    
    
    ### ---- coordinates definition -------- ###

    coords = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"], ndims["glitch"])),
            "noise": np.zeros((ntemps, nwalkers, nleaves_max["noise"], ndims["noise"])),
            }

 
    inds = {
            "glitch": np.zeros((ntemps, nwalkers, nleaves_max["glitch"]), dtype=bool),
            "noise": np.ones((ntemps, nwalkers, nleaves_max["noise"]),  dtype=bool) # we consider one here since we do not do RJ on the noise
        }


    # To shaffle around the inds and being random regarding the inizializations of thw walkers
 
    for nwalk in range(nwalkers):
        for temp in range(ntemps): 
            nl = np.random.randint(nleaves_min["glitch"], nleaves_max["glitch"]+1) 
            inds_tw=np.zeros(nleaves_max["glitch"],dtype=bool) 
            inds_tw[:nl]=True
            np.random.shuffle(inds_tw)
            inds['glitch'][temp,nwalk]=inds_tw
  
    ### ---------  coordinates for the glitches --------- ###

    for nn in range(nleaves_max["glitch"]):

        coords["glitch"][:, :, nn] =priors["glitch"].rvs(size=(ntemps, nwalkers,))

    ### ---------  coordinates for the noise --------- ###
            
    coords["noise"] = priors["noise"].rvs(size=(ntemps, nwalkers,nleaves_max["noise"]))

   ### ---------------------------------------------------- ##
   
    fp = "seven_glitch_from_LPF_with_shapelet_fitting_RJ.h5"

       
    def update_fn(i, res, samp):
        max_it_update=500000
        mem =500

 
        print('---------------------------------------------')
        print("total it", samp.iteration)
        print("max last loglike",np.max(samp.get_log_like()))
        print("min last loglike",np.min(samp.get_log_like()))
        for mm in samp.moves:
            print("move accept",mm.acceptance_fraction[0])
            print("rj \n",samp.rj_acceptance_fraction[0] )
            print("swap \n",samp.swap_acceptance_fraction)

        ##### chains ########

        noise_sampler= samp.get_chain()['noise'][-mem:,0].reshape(-1,2)
        likelihood = samp.get_log_like()[-mem:, 0,:]

        #### reconstruction of the noise ###

        random_oms_par_1 = np.random.choice(noise_sampler[:,0], size=1, replace=False)
        random_tm_par_2 = np.random.choice(noise_sampler[:,1], size=1, replace=False)
                        
        psd_estimated = noisepsd_AET(freqs,Soms_d_in =  random_oms_par_1,Sa_a_in = random_tm_par_2 )

        ## -------------------- plots ------------------ ##

        c = ChainConsumer()
        parameter_labels = ['$OMS$','$TM$']
        c.add_chain(noise_sampler, parameters=parameter_labels, name='noise', color='#6495ed')
        c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
        c.add_marker([(7.9e-12),  (2.4e-15)], marker_style="x", marker_size=500, color='#DC143C')
        fig = c.plotter.plot(figsize=(8,8), legend=True)
        plt.savefig("LISA_noise.png", dpi=300)
        plt.close()

       
        plt.figure()
        plt.loglog(freqs ,  psd_estimated[0] ,'m',label=' A data')
        plt.loglog(freqs , psd_estimated[1] ,'b',label=' T data')
        plt.loglog(freqs ,  psd_AT[0]  ,'k--',label=' A noise model')
        plt.loglog(freqs ,  psd_AT[1]  ,'k--',label=' T noise model')
        plt.legend()
        plt.grid()
        plt.savefig("estimated_noise.png")
        plt.close()


        # Create a plot
        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(len( likelihood )), likelihood )
        plt.xlabel('iter')
        plt.ylabel('Likelihood freq domain')
        plt.title('likelihood evolutions for all the walkers at 0 temperature')
        plt.grid(True)
        plt.show()
        plt.savefig("glitches_likelihood_freq_domain_RJ.png", dpi=300)
        plt.close()

        # ensuring to have at least one glitch #
      
        nleaves = samp.get_nleaves()['glitch'][:,0,:].reshape(-1)


        nleaves_with_0 = np.sum(nleaves == 0)

        print('num of 0 leaves',nleaves_with_0)
  
        fraction0leaves = nleaves_with_0/len(nleaves)
  
        
        if (samp.iteration<max_it_update):
            if fraction0leaves<0.6:
                print('update...')

                ## glitches ##

                item_samp_glitch_no_nans= samp.get_chain()['glitch'][-mem:,0][~np.isnan(samp.get_chain()['glitch'][-mem:,0])].reshape(-1,3)

                gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-15)

                gmm_glitch.fit(item_samp_glitch_no_nans)
               
                #### compute glitch covariance matrix ###

                for move in samp.moves:
                    if hasattr(move, "name") and move.name == "selected covariance":
                        move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)

                ## noise ###
                '''
                noise_sampler= samp.get_chain()['noise'][-mem:,0].reshape(-1,ndims['noise'])
                gmm_noise = GaussianMixture(n_components=1,covariance_type='full', tol=0.001, reg_covar=1e-50)
                gmm_noise.fit(noise_sampler)
                covariances_noise = gmm_noise.covariances_
                    
                for mm, el_cov in zip(samp.moves[3:], covariances_noise):
                    mm.all_proposal['noise'].scale = el_cov
                '''
                c = ChainConsumer()
                parameter_labels = ['$\\tau$', '$ln(dv)$', '$log(\\beta)$']
                item_samp_glitch_no_nans[:,2] =np.log(item_samp_glitch_no_nans[:,2])
                c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
                
                c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)
       
                for ii in range(len(glitch_params)):
                    c.add_marker([glitch_params[ii][0], np.log(np.exp(glitch_params[ii][1])/(2*glitch_params[ii][2])), np.log(glitch_params[ii][2])], \
                    parameters=parameter_labels, marker_style="x", \
                    marker_size=100, color='#DC143C')
        
                fig = c.plotter.plot(figsize=(8,8), legend=True)
                plt.savefig("glitches_corner_plot_RJ.png", dpi=300)
                plt.close()

    
        if (samp.iteration==max_it_update):
            
            for ii, el in enumerate(branch_names):
                print('final update...')
                mem = int(samp.iteration*0.9)
                item_samp_glitch_no_nans= samp.get_chain()[el][-mem:,0][~np.isnan(samp.get_chain()[el][-mem:,0])].reshape(-1,3)
                
                gmm_glitch = GaussianMixture(n_components=nleaves_max['glitch'],covariance_type='full', tol=0.00001, reg_covar=1e-10)

                gmm_glitch.fit(item_samp_glitch_no_nans)

                for move in samp.moves:
                    if hasattr(move, "name") and move.name == "selected covariance":
                        move.update_mean_cov(res.branches, gmm_glitch.means_,  gmm_glitch.covariances_)
                        ##########
                
                parameter_labels = ['$t0$', '$ln(DV)$', '$\\tau$']
                c = ChainConsumer()

                c.add_chain(item_samp_glitch_no_nans, parameters=parameter_labels , name='glitches', color='#6495ed')
                
                c.configure(bar_shade=True, tick_font_size=8, label_font_size=12, max_ticks=8, usetex=True, serif=True)

                for ii in range(len(glitch_params)):
                    c.add_marker([glitch_params[ii][0], np.log(np.exp(glitch_params[ii][1])/(2*glitch_params[ii][2])), np.log(glitch_params[ii][2])], \
                    parameters=parameter_labels, marker_style="x", \
                    marker_size=100, color='#DC143C')
        
                fig = c.plotter.plot(figsize=(8,8), legend=True)
                plt.savefig("glitches_corner_plot_end_RJ.png", dpi=300)

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
            args=[fft_data, df,freqs_cut],
            tempering_kwargs=tempering_kwargs,
            moves=moves,
            rj_moves=True, 
            provide_groups=True,
            nleaves_max=nleaves_max,
            nleaves_min=nleaves_min,
            branch_names=branch_names,
            update_iterations=10,
            update_fn=update_fn,
            nbranches=2,
            vectorize=True,
            backend= HDFBackend(fp))
            

        nsteps =500000
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


