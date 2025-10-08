"""
Fisher Matrix Computation for MBH Parameter Estimation

This module provides functions for computing and loading Fisher matrices.

Functions
---------
build_fish_matrix : Compute Fisher matrix using numerical derivatives
load_fisher_matrix : Load Fisher matrix from file
"""

import numpy as np


def build_fish_matrix(wave_gen, M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref, 
                     window_func=1.0, **kwargs):
    """
    Build Fisher matrix using numerical derivatives.

    Parameters
    ----------
    wave_gen : callable
        Waveform generator function (e.g., MBH_f)
    M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref : float
        MBH parameters
    window_func : array or float, optional
        Gap window function to apply. Default: 1.0 (no gaps)
    **kwargs : dict
        Must contain:
        - 'delta_f': frequency spacing
        - 'delta_t': time spacing  
        - 'PSD': power spectral density
        - 'freq': frequency array
        - 'f_ref': reference frequency
        - 'modes': harmonic modes

    Returns
    -------
    array
        Fisher matrix, shape (11, 11) for 11 MBH parameters

    Notes
    -----
    Uses central finite differences for numerical derivatives.
    Applies gap window in time domain if provided.
    """
    
    # Detect array backend
    try:
        from .waveforms import MBH_f
        from .noise import inner_prod
    except ImportError:
        # Fallback to old imports
        import sys
        import os
        sys.path.insert(0, os.path.dirname(__file__))
        from mbh_utils import MBH_f
        from noise_utils import inner_prod
    
    # Get configuration
    delta_f = kwargs["delta_f"]
    delta_t = kwargs["delta_t"]
    PSD_AET = kwargs["PSD"]
    
    # Detect array module
    if hasattr(PSD_AET[0], '__cuda_array_interface__'):
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            xp = np
    else:
        xp = np

    # Parameter vector
    params = np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref])
    N_params = len(params)

    # Step sizes for numerical derivatives
    steps = np.array([1, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    
    # Compute derivatives
    deriv_vec = []
    params_copy = params.copy()
    
    for j in range(N_params):
        # f(x + h)
        params[j] = params[j] + steps[j]
        h_f_p = MBH_f(wave_gen, *params, **kwargs)
        
        # Apply gap window if provided
        if not isinstance(window_func, (int, float)) or window_func != 1.0:
            h_f_p_t = window_func * xp.fft.irfft(h_f_p / delta_t)
            h_f_p = xp.fft.rfft(h_f_p_t) * delta_t

        # f(x - h)
        params[j] = params[j] - 2 * steps[j]
        h_f_m = MBH_f(wave_gen, *params, **kwargs)
        
        # Apply gap window if provided
        if not isinstance(window_func, (int, float)) or window_func != 1.0:
            h_f_m_t = window_func * xp.fft.irfft(h_f_m / delta_t)
            h_f_m = xp.fft.rfft(h_f_m_t) * delta_t

        # Numerical derivative: df/dx = (f(x+h) - f(x-h)) / (2h)
        deriv_h_f = (h_f_p - h_f_m) / (2 * steps[j])
        deriv_vec.append(deriv_h_f)
        
        # Reset parameters
        params = params_copy.copy()

    # Build Fisher matrix for each channel
    gamma_A, gamma_E = xp.eye(N_params), xp.eye(N_params)

    for i in range(N_params):
        for j in range(i, N_params):
            # Fisher matrix element: Γ_ij = ⟨∂h/∂θ_i | ∂h/∂θ_j⟩
            gamma_A[i, j] = 4 * delta_f * xp.real(
                xp.sum((deriv_vec[i][0] * xp.conjugate(deriv_vec[j][0]) / PSD_AET[0]))
            )
            gamma_E[i, j] = 4 * delta_f * xp.real(
                xp.sum((deriv_vec[i][1] * xp.conjugate(deriv_vec[j][1]) / PSD_AET[1]))
            )

    # Extract diagonal matrices
    gamma_A_diag = xp.diag(xp.diag(gamma_A))
    gamma_E_diag = xp.diag(xp.diag(gamma_E))

    # Subtract off half the diagonal (to correct for double-counting)
    gamma_A = gamma_A - 0.5 * gamma_A_diag
    gamma_E = gamma_E - 0.5 * gamma_E_diag

    # Make symmetric matrices
    gamma_A = gamma_A + gamma_A.T
    gamma_E = gamma_E + gamma_E.T

    # Combine channels
    gamma_AE = gamma_A + gamma_E
    
    return gamma_AE


def load_fisher_matrix(file_path):
    """
    Load Fisher matrix results from .npy file.

    Handles various encoding and pickle compatibility issues.

    Parameters
    ----------
    file_path : str
        Path to .npy file containing Fisher matrix results

    Returns
    -------
    dict
        Dictionary containing Fisher matrix and metadata with keys:
        - 'Fish_Matrix': Fisher matrix array
        - 'Cov_Matrix': Covariance matrix (inverse of Fisher matrix)
        - 'SNR_per_channel': SNR for each channel
        - 'SNR_total': Total SNR
        - 'true_params': Dictionary of injection parameters
        - 'param_names': List of parameter names
        - Additional metadata

    Notes
    -----
    Tries multiple methods to handle different numpy/pickle versions.
    """
    import pickle

    # Method 1: Standard numpy load with .item()
    try:
        FM_results = np.load(file_path, allow_pickle=True).item()
        print("✓ Loaded with np.load().item()")
        return FM_results
    except Exception as e:
        print(f"Method 1 failed: {e}")

    # Method 2: Load with encoding='latin1'
    try:
        FM_results = np.load(file_path, allow_pickle=True, encoding='latin1').item()
        print("✓ Loaded with encoding='latin1'")
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

    # Method 4: Pickle with encoding
    try:
        with open(file_path, 'rb') as f:
            FM_results = pickle.load(f, encoding='latin1')
        print("✓ Loaded with pickle.load(encoding='latin1')")
        return FM_results
    except Exception as e:
        print(f"Method 4 failed: {e}")

    raise RuntimeError(f"Could not load Fisher matrix from {file_path} using any method")
