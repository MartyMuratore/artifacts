import numpy as np
from lisatools.utils.constants import PC_SI,YRSID_SI
from bbhx.waveformbuild import BBHWaveformFD
import scipy
use_gpu = False
if use_gpu:
    import cupy as cp
    xp = cp
else:
    xp = np

def MBH_f(wave_gen, M,q,a1,a2,inc,dist_Gpc,phi_ref,lam,beta,psi,t_ref, **kwargs):
    """
    Code to generate massive black holes. Change of parametrisation to M, q and distance
    in Gpc. Outputs TDI A, E T for MBH. 
    """
    freq = kwargs['freq']
    f_ref = kwargs['f_ref'] 
    modes = kwargs['modes']
    delta_t = kwargs['delta_t']
    m1 = q*M/(1.0 + q)
    m2 = M/(1.0 + q)
    dist_m = dist_Gpc * 1e9 * PC_SI
    MBH_AET = wave_gen(m1,m2,a1,a2,dist_m,phi_ref,f_ref,inc,lam,beta,psi,t_ref, freqs=freq, modes=modes, 
                       direct=False, fill=True, squeeze=True, 
                       length=len(freq))[0][0:2]

    
    return MBH_AET/delta_t



def build_fish_matrix(wave_gen, M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref, window_func = 1.0, **kwargs):
    """
    Build the fisher matrix. Inputs parameters, outputs fisher matrix (NOT covariance matrix)
    """
    delta_f = kwargs["delta_f"]
    PSD_AET = kwargs["PSD"] 

    params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref])
    N_params = len(params)

    steps = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6]) # Steps for numerical derivative. Seem "fine"
    N_params = len(steps) 
    deriv_vec = []
    params_copy = params.copy()
    for j in range(N_params):
        params[j] = params[j] + steps[j] # this is the f(x + h) step
        h_f_p = MBH_f(wave_gen,*params, **kwargs) 
        h_f_p_t = window_func*xp.fft.irfft(h_f_p) # Apply gap if required
        h_f_p = xp.fft.rfft(h_f_p_t)

        params[j] = params[j] - 2 * steps[j] # this is the f(x - h) step
        h_f_m = MBH_f(wave_gen,*params, **kwargs)
        h_f_m_t = window_func*xp.fft.irfft(h_f_m) # Apply gap if required
        h_f_m = xp.fft.rfft(h_f_m_t)

        deriv_h_f = (h_f_p - h_f_m) / (2*steps[j]) # compute derivative
        deriv_vec.append(deriv_h_f)
        params = params_copy # reset parameters

    # For build matrices to place values
    gamma_A, gamma_E = xp.eye(N_params), xp.eye(N_params)

    for i in range(N_params):
        for j in range(i,N_params):
            gamma_A[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][0]*xp.conjugate(deriv_vec[j][0]) / (PSD_AET[0]))))
            gamma_E[i,j] = 4*delta_f * xp.real(xp.sum((deriv_vec[i][1]*xp.conjugate(deriv_vec[j][1]) / (PSD_AET[1]))))

    # Build diagonal matrices 
    gamma_A_diag = xp.diag(xp.diag(gamma_A)) 
    gamma_E_diag = xp.diag(xp.diag(gamma_E))

    # Subtract off first element of diagonal matrix
    gamma_A = gamma_A - 0.5*gamma_A_diag
    gamma_E = gamma_E - 0.5*gamma_E_diag

    # Build symmetric matrix. Notice that diagonal is twice that of gamma_(AE) [restoring consistency]
    gamma_A = gamma_A + gamma_A.T
    gamma_E = gamma_E + gamma_E.T

    # Compute joint FM over A and E. 
    gamma_AE = gamma_A + gamma_E 
    return gamma_AE



def load_fisher_matrix(file_path):
    """
    Robustly load Fisher matrix results from .npy file.
    Handles various encoding and pickle compatibility issues.
    """
    import pickle

    # Method 1: Standard numpy load with .item()
    try:
        FM_results = np.load(file_path, allow_pickle=True).item()
        print("✓ Loaded with np.load().item()")
        return FM_results
    except Exception as e:
        print(f"Method 1 failed: {e}")

    # Method 2: Load and access with [()] indexing
    
    # Method 1: Standard numpy load with .item()
    try:
        FM_results = np.load(file_path, allow_pickle=True).item()
        print("✓ Loaded with np.load().item()")
        return FM_results
    except Exception as e:
        print(f"Method 1 failed: {e}")
    
    # Method 2: Load and access with [()] indexing
    try:
        FM_results = np.load(file_path, allow_pickle=True)[()]
        print("✓ Loaded with np.load()[()]")
        return FM_results
    except Exception as e:
        print(f"Method 2 failed: {e}")
    
    # Method 3: Direct pickle load
    try:
        with open(file_path, 'rb') as f:
            FM_results = pickle.load(f)
        print("✓ Loaded with pickle.load()")
        return FM_results
    except Exception as e:
        print(f"Method 3 failed: {e}")
    
    raise IOError(f"Could not load Fisher matrix from {file_path}. Please regenerate the file by running mcmc_MBH.py")

