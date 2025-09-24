
# I import here utilities to be used within the code. 
import numpy as np
from bbhx.utils.constants import *
from bbhx.utils.transform import *


## This is to set the GPU to use. The code works also without GPU!

try:
    import cupy as xp
    # set GPU device
    xp.cuda.runtime.setDevice(0)
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

## if not GPU are used  
use_gpu = False

if use_gpu is False:
    xp = np

## -----  definition of general functions to be used on the analytical model ---- ###

def HeavisideTheta(x, xp=None):

    if xp is None:
        xp = np
    
    squeeze = True if x.ndim == 1 else False

    x = xp.atleast_1d(x)

    out = 1.0 * (x >= 0.0) # I set to zero all the case when the argument x is less then zero

    if squeeze:
        out = out.squeeze()

    return out

# A, E and T definition

def AET(X, Y, Z):
        return (
            (Z - X) / np.sqrt(2.0),
            (X - 2.0 * Y + Z) / np.sqrt(6.0),
            (X + Y + Z) / np.sqrt(3.0))

## -----    Glitch waveform definition time and frequency domains --------##

######### ---------- time domain TDI first generation   ---------- ###### 


# In this part of the code I define the function for computing the glitch in the link 12 

#Note: use this expression if you want to estimate the natural logarit of the glitch amplitude for single exponential model
def tdi_glitch_link12(t, t0=1.9394221536001746,  Deltav=np.log(2.22616837*10**(-11)), tau=0.79357148 ,T=8.3, co = C_SI, xp=None):    

    if xp is None:
        xp = np

    tau = xp.atleast_1d(tau)
    t0 = xp.atleast_1d(t0)
    Deltav = xp.atleast_1d(Deltav)

    assert tau.shape == t0.shape == Deltav.shape

    ## The glitch expression is divided in multiple parts to account for overflow when the exponential is a big number
 
    out = xp.zeros((3, len(t0), len(t)))

    mask1 =  t[None,:] >= t0[:,None]
   
    expression1 = (xp.where(mask1,HeavisideTheta(t[None,:] - t0[:,None]),0) * (-1 + xp.where(mask1,np.exp( (- t[None,:] + t0[:,None] ) / tau[:,None]),0) + 
                    t[None,:] * xp.where(mask1,np.exp( (- t[None,:] + t0[:,None] ) / tau[:,None]),0) /  tau[:,None] - xp.where(mask1,np.exp( (- t[None,:] + t0[:,None] ) / tau[:,None]),0)*   t0[:,None] /  tau[:,None]))

    mask2 = t[None,:] >= t0[:,None] + 4 * T

    expression2 =( xp.where(mask2,HeavisideTheta(t[None,:] - t0[:,None] - 4 * T ),0) * (1 - xp.where(mask2,np.exp( (- t[None,:] + t0[:,None] + 4*T ) / tau[:,None]),0) 
                    - t[None,:] * xp.where(mask2,np.exp( (- t[None,:] + t0[:,None] + 4*T ) / tau[:,None]),0) /  tau[:,None]    + 4 * T * xp.where(mask2,np.exp( (- t[None,:] + t0[:,None] + 4*T ) / tau[:,None]),0) /  tau[:,None] 
                     +  t0[:,None] * xp.where(mask2,np.exp( (- t[None,:] + t0[:,None] + 4*T ) / tau[:,None]),0) /  tau[:,None]  ))


    mask3 =  t[None,:] >= t0[:,None] + 3 *T

    expression3 =(xp.where(mask3, HeavisideTheta(t[None,:] - t0[:,None] - 3 *T),0) 
                 *  (-1 + xp.where(mask3,np.exp( (- t[None,:] + t0[:,None] + 3 *T) / tau[:,None]),0) *(1 +  t[None,:] /tau[:,None] - 3 * T /tau[:,None] - t0[:,None] /tau[:,None]) ))
             

    mask4 = t[None,:] >= t0[:,None] + T

    expression4 = (xp.where(mask4, HeavisideTheta(t[None,:] - t0[:,None] - T) ,0) * 
                  (1 + xp.where(mask4,np.exp( (- t[None,:] + t0[:,None] + T) / tau[:,None]),0) *(-1 -  t[None,:] /tau[:,None] + T /tau[:,None] + t0[:,None] /tau[:,None])  ))
                  
    
    ## Analytical expression X,Y,Z first generation

    tdiX1link12 = np.exp(Deltav[:,None])/co*  (expression1 +  expression2)
    tdiY1link12 =  2* np.exp(Deltav[:,None])/co* (expression3  + expression4)
    tdiZ1link12 = xp.zeros_like(tdiX1link12)

    out[0,] = tdiX1link12 
    out[1, ] = tdiY1link12
    out[2, ] = tdiZ1link12
 
    return out

#Note: use this expression if you want to estimate the natural logarit of the glitch amplitude for double exponential model 
def tdi_2exp_glitch_link12(t, t0=950,  Deltav=np.log(2.22616837*10**(-11)), tau1=10, tau2=11 ,T=8.3, co = C_SI, xp=None):    

    if xp is None:
        xp = np

    tau1 = xp.atleast_1d(tau1)
    tau2 = xp.atleast_1d(tau2)
    t0 = xp.atleast_1d(t0)
    Deltav = xp.atleast_1d(Deltav)

    assert tau1.shape == tau2.shape == t0.shape == Deltav.shape
 
    out = xp.zeros((3, len(t0), len(t)))
 
    mask1 =  t[None,:] >= t0[:,None]
    mask2 =  t[None,:] >= t0[:,None] + 8*T
    mask3 =  t[None,:] >= t0[:,None] + 4*T
   
    tdiX1link12 =( xp.exp(Deltav[:,None])/(co*(tau1[:,None] - tau2[:,None])) *
     ((tau1[:,None] - xp.where(mask1,xp.exp((-t + t0)/tau1[:,None]),0)*tau1[:,None] + 
            (-1 +xp.where(mask1,np.exp((-t + t0)/tau2[:,None]),0))*tau2[:,None])* xp.where(mask1,HeavisideTheta(t - t0),0) + 
           (tau1[:,None] - xp.where(mask2,xp.exp((-t + 8*T + t0)/tau1[:,None]),0)*tau1[:,None] + 
           (-1 +xp.where(mask2,xp.exp((-t + 8*T + t0)/tau2[:,None]),0))*tau2[:,None])* xp.where(mask2,HeavisideTheta(t - 8*T - t0),0) 
            - 2*(tau1[:,None] - xp.where(mask3,xp.exp((-t + 4*T + t0)/tau1[:,None]),0)*tau1[:,None]  
         +  (-1 +xp.where(mask3,xp.exp((-t + 4*T + t0)/tau2[:,None]),0))*tau2[:,None])* xp.where(mask3,HeavisideTheta(t - 4*T - t0),0)))

    mask4 =  t[None,:] >= t0[:,None] + 7*T
    mask5 =  t[None,:] >= t0[:,None] + 5*T
    mask6 =  t[None,:] >= t0[:,None] + 3*T
    mask7 =  t[None,:] >= t0[:,None] + T


    tdiY1link12 = ( - 2* np.exp(Deltav[:,None])/(co*(tau1[:,None] - tau2[:,None])) * ( ((tau1[:,None] -xp.where(mask7,xp.exp((-t + 7*T + t0)/tau1[:,None]),0)*tau1[:,None]  
      +  (-1 + xp.where(mask7,xp.exp((-t + 7*T + t0)/tau2[:,None]),0))*tau2[:,None])* xp.where(mask4,HeavisideTheta(t - 7*T - t0),0))  
        + ((-1 + xp.where(mask5,xp.exp((-t + 5*T + t0)/tau1[:,None]),0))*tau1[:,None]+ tau2[:,None]  
        -   xp.where(mask5,xp.exp((-t + 5*T + t0)/tau2[:,None]),0)*tau2[:,None])* xp.where(mask5,HeavisideTheta(t - 5*T - t0),0)
     +  ((-1 + xp.where(mask6,xp.exp((-t + 3*T + t0)/tau1[:,None]),0))*tau1[:,None] + tau2[:,None]  
     -     xp.where(mask6,xp.exp((-t + 3*T + t0)/tau2[:,None]),0)*tau2[:,None])*  xp.where(mask6,HeavisideTheta(t - 3*T - t0) ,0) 
      +  (tau1[:,None] -  xp.where(mask7,xp.exp((-t + T + t0)/tau1[:,None]),0)*tau1[:,None]
      +     (-1 + xp.where(mask7,xp.exp((-t + T + t0)/tau2[:,None]),0))*tau2[:,None])*xp.where(mask7,HeavisideTheta(t - T - t0),0)))


    tdiZ1link12 = xp.zeros_like(tdiX1link12)

    out[0,] = tdiX1link12 
    out[1, ] = tdiY1link12
    out[2, ] = tdiZ1link12
 
    return out

#### --------- Frequency domain for TDI first and second generation ------ ###

## Single exponential glitch model 
def tdi_glitch_link12_frequency_domain(freq, t0=1.9394221536001746,  Deltav=2.22616837*10**(-11), tau=0.79357148 ,T=8.3,co = C_SI, tdi_channel='first_gen', xp=None):    

    if xp is None:
        xp = np

    tau = xp.atleast_1d(tau)
    t0 = xp.atleast_1d(t0)
    Deltav = xp.atleast_1d(Deltav)

    assert tau.shape == t0.shape == Deltav.shape
 
    out = xp.zeros((3, len(t0), len(freq)), dtype=xp.cdouble)

    ## Glitch amplitude
   
    Deltanuh = - 1/(1j*co*2*np.pi*freq[None,:]) * np.exp(-1j * t0[:,None] *2*np.pi*freq[None,:]) * np.exp(Deltav[:,None]) /(-1j + tau[:,None]* 2*np.pi*freq[None,:])**2

    ## Transfer fuction 

    if tdi_channel == 'first_gen':

        TFX_single_glich_tm12 = (-1 + np.exp(-4 * 1j* T* 2*np.pi*freq[None,:]))
        TFY_single_glich_tm12 = 4 * 1j * np.exp(-2 * 1j* T *  2*np.pi*freq[None,:])*np.sin(T*2*np.pi*freq[None,:])
        TFZ_single_glich_tm12 = xp.zeros_like(TFY_single_glich_tm12)

        ## this is the model of the glitch in frequency domain where it appears as a low pass filter

        out[0,] = TFX_single_glich_tm12*Deltanuh
        out[1, ] =TFY_single_glich_tm12*Deltanuh
        out[2, ] =TFZ_single_glich_tm12*Deltanuh

    elif tdi_channel == 'second_gen':

        TFX_single_glich_tm12 = xp.exp(-8 *1j * T *2*np.pi*freq[None,:])* (-1 + xp.exp(4 * 1j* T* 2*np.pi*freq[None,:]))**2
        TFY_single_glich_tm12 = 8 * xp.exp(-4 * 1j*T *2*np.pi*freq[None,:]) * xp.sin(2 * T *  2*np.pi*freq[None,:])*xp.sin(T*2*np.pi*freq[None,:])
        TFZ_single_glich_tm12 = xp.zeros_like(TFY_single_glich_tm12)

        out[0,] = TFX_single_glich_tm12*Deltanuh
        out[1, ] =TFY_single_glich_tm12*Deltanuh
        out[2, ] =TFZ_single_glich_tm12*Deltanuh

    else:
        raise ValueError(f"Unknown TDI channel: {tdi_channel}. Valid options are 'first_gen' or 'second_gen'.")

    return out

## Double exponential glitch model 
##Note the natural logartim of the amplitude is estimated
def tdi_2expglitch_link12_frequency_domain(freq, t0=1.9394221536001746,  Deltav=np.log(2.22616837*10**(-11)), tau1=0.79357148,tau2=0.79357148 ,T=8.3,co = C_SI, tdi_channel='second_gen', xp=None):    

    if xp is None:
        xp = np

    tau1 = xp.atleast_1d(tau1)
    tau2 = xp.atleast_1d(tau2)
    t0 = xp.atleast_1d(t0)
    Deltav = xp.atleast_1d(Deltav)

    assert tau1.shape == tau2.shape == t0.shape == Deltav.shape
 
    out = xp.zeros((3, len(t0), len(freq)), dtype=xp.cdouble)
   
    Deltanuh = 1/(1j*co*2*np.pi*freq[None,:]) * np.exp(-1j * t0[:,None] *2*np.pi*freq[None,:]) * np.exp(Deltav[:,None]) /((1+1j * tau1[:,None]* 2*np.pi*freq[None,:])*(1 +1j * tau2[:,None]* 2*np.pi*freq[None,:]))

    ## this is the transfer fuction 

    if tdi_channel == 'first_gen':

        TFX_single_glich_tm12 = (-1 + np.exp(-4 * 1j* T* 2*np.pi*freq[None,:]))
        TFY_single_glich_tm12 = 4 * 1j * np.exp(-2 * 1j* T *  2*np.pi*freq[None,:])*np.sin(T*2*np.pi*freq[None,:])
        TFZ_single_glich_tm12 = xp.zeros_like(TFY_single_glich_tm12)

        ## this is the model of the glitch in frequency domain where it appears as a low pass filter

        out[0,] = TFX_single_glich_tm12*Deltanuh
        out[1, ] =TFY_single_glich_tm12*Deltanuh
        out[2, ] =TFZ_single_glich_tm12*Deltanuh

    elif tdi_channel == 'second_gen':

        TFX_single_glich_tm12 = xp.exp(-8 *1j * T *2*np.pi*freq[None,:])* (-1 + xp.exp(4 * 1j* T* 2*np.pi*freq[None,:]))**2
        TFY_single_glich_tm12 = 8 * xp.exp(-4 * 1j*T *2*np.pi*freq[None,:]) * xp.sin(2 * T *  2*np.pi*freq[None,:])*xp.sin(T*2*np.pi*freq[None,:])
        TFZ_single_glich_tm12 = xp.zeros_like(TFY_single_glich_tm12)

        out[0,] = TFX_single_glich_tm12*Deltanuh
        out[1, ] =TFY_single_glich_tm12*Deltanuh
        out[2, ] =TFZ_single_glich_tm12*Deltanuh

    else:
        raise ValueError(f"Unknown TDI channel: {tdi_channel}. Valid options are 'first_gen' or 'second_gen'.")

    return out

### In this part of the code I define the function for computing the shapelet in the link 12 for n0=1

######### ---------- time domain TDI first generation   ---------- ###### 

def tdi_shapelet_link12(t, tau=1.9394221536001746,  Deltav=2.22616837*10**(-11), beta=0.79357148 ,T=8.3, co = C_SI, xp=None): 

    if xp is None:
       xp = np

    tau = xp.atleast_1d(tau)
    beta = xp.atleast_1d(beta)
    Deltav = xp.atleast_1d(Deltav)

    assert tau.shape == beta.shape == Deltav.shape
 
    out = xp.zeros((3, len(beta), len(t)))


    # without the diract delta 
    tdiX1link12 =  (-2*(np.exp(Deltav[:,None])*(beta[:,None] - 
              np.exp((-t[None,:] + tau[:,None])/beta[:,None])*(t[None,:] + beta[:,None] -tau[:,None]))*
            HeavisideTheta(t[None,:] - tau[:,None]) - 
           np.exp(Deltav[:,None])*(beta[:,None] + 
              np.exp((-t[None,:] + 4*T + tau[:,None])/beta[:,None])*
              (-t[None,:] + 4*T -beta[:,None] +tau[:,None]))*
            HeavisideTheta(t[None,:] - 4*T - tau[:,None])))/co
 
    tdiY1link12 = (-4*(np.exp(Deltav[:,None])*(beta[:,None] + 
              np.exp((-t + 3*T + tau[:,None])/beta[:,None])*
               (-t + 3*T - beta[:,None] + tau[:,None]))*
            HeavisideTheta(t - 3*T -tau[:,None]) - 
           np.exp(Deltav[:,None])*(beta[:,None] + 
              np.exp((-t + T + tau[:,None])/beta[:,None])*
               (-t + T - beta[:,None] + tau[:,None]))*
            HeavisideTheta(t - T -tau[:,None])))/co
  

    tdiZ1link12 = xp.zeros_like(tdiX1link12)
    
    out[0,] = tdiX1link12 
    out[1, ] = tdiY1link12
    out[2, ] = tdiZ1link12


    return out

#### --------- Frequency domain for TDI first and second generation ------ ###

def tdi_shapelet_link12_frequency_domain(freq, tau=0.79357148,  Deltav=2.22616837*10**(-11), beta=1.9394221536001746 ,co = C_SI,T=8.3, tdi_channel='second_gen', xp=None):    

    if xp is None:
        xp = np

    tau = xp.atleast_1d(tau)
    beta = xp.atleast_1d(beta)
    Deltav = xp.atleast_1d(Deltav)

    assert tau.shape == beta.shape == Deltav.shape
 
    out = xp.zeros((3, len(beta), len(freq)), dtype=np.cdouble)
   
    Deltanus = - 2*xp.exp(Deltav[:,None])* beta[:,None]/(1j*co*2*np.pi*freq[None,:]) * xp.exp(-1j * tau[:,None] *2*np.pi*freq[None,:])/(-1j + beta[:,None]* 2*np.pi*freq[None,:])**2

    ## this is the transfer fuction 

    if tdi_channel == 'first_gen':

        TFX1_single_glich_tm12 = (-1 + xp.exp(-4 * 1j* T* 2*np.pi*freq[None,:]))
        TFY1_single_glich_tm12 = 4 * 1j * xp.exp(-2 * 1j* T *  2*np.pi*freq[None,:])*xp.sin(T*2*np.pi*freq[None,:])
        TFZ1_single_glich_tm12 = xp.zeros_like(TFY1_single_glich_tm12)

        out[0,] =  TFX1_single_glich_tm12*Deltanus
        out[1, ] =TFY1_single_glich_tm12*Deltanus
        out[2, ] =TFZ1_single_glich_tm12*Deltanus
    
    elif tdi_channel == 'second_gen':

        TFX_single_glich_tm12 = xp.exp(-8 *1j * T *2*np.pi*freq[None,:])* (-1 + xp.exp(4 * 1j* T* 2*np.pi*freq[None,:]))**2
        TFY_single_glich_tm12 = 8 * xp.exp(-4 * 1j*T *2*np.pi*freq[None,:]) * xp.sin(2 * T *  2*np.pi*freq[None,:])*xp.sin(T*2*np.pi*freq[None,:])
        TFZ_single_glich_tm12 = xp.zeros_like(TFY_single_glich_tm12)


        out[0,] = TFX_single_glich_tm12*Deltanus
        out[1, ] =TFY_single_glich_tm12*Deltanus
        out[2, ] =TFZ_single_glich_tm12*Deltanus

    else:
        raise ValueError(f"Unknown TDI channel: {tdi_channel}. Valid options are 'first_gen' or 'second_gen'.")


    return out


## --------  These are the functions used to combine multiple glitches or shapelets  ----- ##

# Time domain multiple glitches with single exponentials

def combine_single_exp_glitch_link12(t_in, params, T=8.3,xp=None):
    if xp is None:
       xp = np
  
    t0 = xp.atleast_1d(params[:, 0])
    Deltav = xp.atleast_1d(params[:, 1])
    tau = xp.atleast_1d(params[:, 2])

    assert t0.shape == Deltav.shape == tau.shape
    

    out =  tdi_glitch_link12(t_in, t0=t0,  Deltav=Deltav, tau=tau,T=T, xp=xp).sum(axis=1, keepdims=True)  # *params -> a, b, c

    return out

# Frequency domain multiple glitches with single exponentials

def combine_single_exp_glitch_link12_frequency_domain(freq, params, T=8.3 , tdi_channel='second_gen', xp=None):
    if xp is None:
       xp = np
  
    t0 = xp.atleast_1d(params[:, 0])
    Deltav = xp.atleast_1d(params[:, 1])
    tau = xp.atleast_1d(params[:, 2])

    assert t0.shape == Deltav.shape == tau.shape
    
    out =  tdi_glitch_link12_frequency_domain(freq, t0=t0,  Deltav=Deltav, tau=tau,T=T, tdi_channel=tdi_channel, xp=xp).sum(axis=1, keepdims=True)  # *params -> a, b, c

    return out

# Frequency domain multiple shapelet

def combine_shapelet_link12_frequency_domain(freq, params, T=8.3,tdi_channel='second_gen',xp=None):
    if xp is None:
       xp = np

    tau = xp.atleast_1d(params[:, 0])
    beta = xp.atleast_1d(params[:, 2])
    Deltav = xp.atleast_1d(params[:, 1])

    assert tau.shape == Deltav.shape == beta.shape

    out = tdi_shapelet_link12_frequency_domain(freq, tau=tau,Deltav=Deltav, beta=beta  ,T=T,co = C_SI, tdi_channel=tdi_channel, xp=xp).sum(axis=1, keepdims=True)  # *params -> a, b, c

    return out

#### ------- This part of the script is run only if this .py file is run directly. Here the function are evaluated and the plots are made #####

if __name__ == "__main__":
  
    ## defining constant to evaluate the glitches/shapelet

    dt = 0.25
    Tobs = 1/12. *YRSID_SI
    N = int(Tobs / dt)
    Tobs = dt * N
    t_in = np.arange(N) * dt

    freqs = np.fft.rfftfreq(N, dt)  # fs =1/dt
    freqs[freqs == 0] = 1e-50

    fmin = 2.5e-5
    fmax = 1e-1
    frequencymask = (freqs > fmin) & (freqs < fmax) # remove ALL the wiggles CAREFULL: we MUST find a way to include them
    freqs_cut = np.array(freqs[frequencymask])

    ###----------  Evaluating the functions for the glitch  --------- ###

    # --- Time domain two exponential glitch model --- #
   
    Xglitch , Yglitch ,Zglitch = tdi_2exp_glitch_link12(t_in, t0=940000,  Deltav=np.log(2.22616837*10**(-11)), tau1=10, tau2=11 ,T=8.3, co = C_SI, xp=None).squeeze()

    Aglitch,Eglitch,Tglitch = AET(Xglitch , Yglitch ,Zglitch)

    # --- Frequency domain numerical computation --- # 

    Aglitch_fft = np.fft.rfft(Aglitch) * dt # TD glitch
    Eglitch_fft = np.fft.rfft(Eglitch) * dt # TD glitch
    Tglitch_fft = np.fft.rfft(Tglitch) * dt # TD glitch

    # --- Frequency domain analytical computation --- # 

    Xglitchw , Yglitchw ,Zglitchw = tdi_2expglitch_link12_frequency_domain(freqs_cut,t0=940000,  Deltav=np.log(2.22616837*10**(-11)), tau1=10, tau2=11).squeeze()

    Aglitchw,Eglitchw,Tglitchw = AET(Xglitchw , Yglitchw ,Zglitchw)
    
    # --- combination of glitch -- #

    glitch_inj_params  = [[2.50000000e+05, 4.32944603e-13, 3.72026813e+01],
       [2.50000000e+06, 5.81021172e-11, 1.39633843e+01]]

    X_2_glitchw , Y_2_glitchw ,Z_2_glitchw = combine_single_exp_glitch_link12_frequency_domain(freqs, glitch_inj_params, tdi_channel='first_gen', T=8.3, xp=xp).squeeze()
    A_2_glitchw,E_2_glitchw,T_2_glitchw = AET(X_2_glitchw , Y_2_glitchw ,Z_2_glitchw)


    ###----------  Evaluating the functions for the shapelet  --------- ###

    Xshapelet , Yshapelet ,Zshapelet = tdi_shapelet_link12(t_in, tau=500,  Deltav=np.log(5*10**(-7)), beta=1e2).squeeze()

    Ashapelet,Eshapelet,Tshapelet = AET(Xshapelet , Yshapelet ,Zshapelet)

    # --- Frequency domain numerical  computation --- # 

    Ashapelet_fft = np.fft.rfft(Ashapelet) * dt 
    Eshapelet_fft = np.fft.rfft(Eshapelet) * dt 
    Tshapelet_fft = np.fft.rfft(Tshapelet) * dt 

    # --- Frequency domain analytical  computation --- # 

    Xshapeletw , Yshapeletw ,Zshapeletw = tdi_shapelet_link12_frequency_domain(freqs,tau=500,  Deltav=5*10**(-7), beta=1e2).squeeze()

    Ashapeletw,Eshapeletw,Tshapeletw = AET(Xshapeletw , Yshapeletw ,Zshapeletw)
   
    # --- combination of shapelet -- #

    shap_inj_params  = [[2.50000000e+05, 4.32944603e-13/(2*3.72026813e+01), 3.72026813e+01],
       [2.50000000e+06, 5.81021172e-11/(2*1.39633843e+01), 1.39633843e+01]]

    X_2_shapeletw, Y_2_shapeletw ,Z_2_shapeletw = combine_shapelet_link12_frequency_domain(freqs, shap_inj_params, tdi_channel='second_gen', xp=None).squeeze()
    A_2_shapeletw,E_2_shapeletw,T_2_shapeletw = AET(X_2_shapeletw , Y_2_shapeletw ,Z_2_shapeletw)

    # ---- Examples of Plots that can be made ------ # 

    import matplotlib.pyplot as plt

    plt.figure()
    plt.loglog(freqs[frequencymask],np.abs(A_2_shapeletw[frequencymask]), 'r', label='2 shapelet TDI A')
    plt.loglog(freqs[frequencymask],np.abs(A_2_glitchw[frequencymask]), 'y--', label='2 glitch TDI A')
    plt.ylabel('Amplitude')
    plt.xlabel('frequency [Hz]')
    plt.grid()
    plt.legend()
    plt.savefig("Frequency_domain_waveform_glitch_vs_shapelet.png")

  


