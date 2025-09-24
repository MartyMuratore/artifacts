import os
import matplotlib.pyplot as plt
import corner
import cupy as cp
import numpy as np

def get_sinthetic_noise(freqs,df,psd):

### we generate here a syntetic noise in frequency from a given psd. The df = 1/T = 1/(N* dt), with N the total duration of the time series
# dt the time interval b/w two samples. In this case df can be computed directly from the frequency axis of the psd considered

# the input are the psd and the frequency resolution. if a psd is not given the default is psd A from lisatools

# the process consider gaussian noise 


# see https://arxiv.org/pdf/2308.01056
    
    
    std_dev = np.sqrt(psd/(4 * df))

    noise_ftt = std_dev *( np.random.normal(loc = 0 , scale = 1 , size = len(psd) ) +1j *np.random.normal(loc = 0 , scale = 1 , size = len(psd) ))
    
    return noise_ftt


def get_sinthetic_psd(df,frequency_series):

## to generate sinthetic pds from a noise fft

   # see power spectral density wikipedia (https://en.wikipedia.org/wiki/Spectral_density#Power_spectral_density). 
   # the 2 here is for the one sided psd

   sinthetic_psd = 2 *df * np.abs(frequency_series)**2

   return sinthetic_psd



## PSD = 2 / T |x_tilde|^2

#  { |x_tilde|^2 } = { Re[x_tilde]^2 } + { Im[x_tilde]^2 }

# but  { Re[x_tilde]^2 } = { Im[x_tilde]^2 }   -->   { Re[x_tilde]^2 }  = PSD T /4  can be a rought check of the frequency series 
