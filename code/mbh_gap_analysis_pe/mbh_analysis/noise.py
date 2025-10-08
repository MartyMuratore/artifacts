"""
Noise Utilities for LISA Data Analysis

This module provides functions for PSD handling, noise generation, and inner products.

Functions
---------
inner_prod : Compute noise-weighted inner product
load_psd_from_file : Load PSD from file and create interpolator
generate_colored_noise : Generate colored Gaussian noise
pad_to_length : Pad arrays to specified length
"""

import numpy as np
import scipy
import os


def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product using BBHx's standard normalization.

    Uses: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]

    Parameters
    ----------
    signal_1_f : array-like
        First signal in frequency domain
    signal_2_f : array-like
        Second signal in frequency domain
    PSD : array-like
        Power spectral density Sn(f)
    delta_f : float
        Frequency spacing (Hz)
    xp : module, optional
        Array module (numpy or cupy). Default: numpy

    Returns
    -------
    float
        Inner product value ⟨signal_1|signal_2⟩
    """
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))


def load_psd_from_file(psd_file, xp=np):
    """
    Load PSD from file and return interpolation function.

    Parameters
    ----------
    psd_file : str
        Path to PSD file (.npy format)
    xp : module, optional
        Array module (numpy or cupy). Default: numpy

    Returns
    -------
    callable
        PSD interpolation function psd(f) that returns PSD at frequency f

    Notes
    -----
    File format: np.vstack((freqs, Sn)).T
    Output is clipped to reasonable bounds to avoid numerical issues.
    """

    psd_in = np.load(psd_file).T
    freqs, values = psd_in[0], np.atleast_2d(psd_in[1:])

    # Convert to cupy if needed
    freqs = xp.asarray(freqs)
    values = xp.asarray(values)

    backend = "cpu" if xp is np else "gpu"
    print(f"Using {backend} backend for PSD interpolation")

    # Get bounds for clipping
    min_psd = np.min(values[:, freqs < 1e-2], axis=1)
    max_psd = np.max(values, axis=1)
    
    # Create interpolator
    psd_interp = scipy.interpolate.CubicSpline(freqs, values, axis=1, extrapolate=True)

    def psd_clipped(f, **kwargs):
        """Interpolated PSD with clipping."""
        f = xp.clip(f, 0.00001, 1.0)
        out = xp.array([
            xp.clip(xp.atleast_2d(psd_interp(f))[i], min_psd[i], max_psd[i])
            for i in range(len(values))
        ])
        return xp.squeeze(out)

    return psd_clipped


def generate_colored_noise(variance_noise_AET, delta_t, seed=0, window_function=None, return_time_domain=False):
    """
    Generate colored Gaussian noise with specified variance per frequency bin.

    Parameters
    ----------
    variance_noise_AET : list or array
        Variance per frequency bin for each channel
        Shape: (N_channels, N_freq) or list of arrays
    delta_t : float
        Time sampling interval (seconds)
    seed : int, optional
        Random seed. Default: 0
    window_function : array-like, optional
        Time-domain window to apply. Default: None (no window)
    return_time_domain : bool, optional
        If True, return noise in time domain. Default: False (frequency domain)

    Returns
    -------
    array or list
        Colored noise in frequency or time domain
        Shape: (N_channels, N_freq) or (N_channels, N_time)

    Notes
    -----
    Generates Gaussian noise with specified variance and applies optional windowing.
    """
    
    # Detect array module
    if hasattr(variance_noise_AET[0], '__cuda_array_interface__'):
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            xp = np
    else:
        xp = np

    # Set random seed
    if xp is np:
        np.random.seed(seed)
    else:
        xp.random.seed(seed)

    N_channels = len(variance_noise_AET)
    noise_freq = []

    for k in range(N_channels):
        # Generate white noise in frequency domain
        variance = variance_noise_AET[k]
        N_freq = len(variance)
        
        # Complex Gaussian noise
        noise_real = xp.random.normal(0, xp.sqrt(variance / 2), N_freq)
        noise_imag = xp.random.normal(0, xp.sqrt(variance / 2), N_freq)
        noise_f = noise_real + 1j * noise_imag
        
        noise_freq.append(noise_f)

    if window_function is not None:
        # Apply window in time domain
        return _apply_window_to_noise(noise_freq, delta_t, window_function, xp, return_time_domain)
    elif return_time_domain:
        # Just transform to time domain
        return xp.asarray([xp.fft.irfft(noise_freq[k] / delta_t) for k in range(N_channels)])
    else:
        return xp.asarray(noise_freq)


def _apply_window_to_noise(noise_freq, delta_t, window_function, xp, return_time_domain):
    """Apply time-domain window to noise."""
    N_channels = len(noise_freq)
    
    # Transform to time domain
    noise_time = [xp.fft.irfft(noise_freq[k] / delta_t) for k in range(N_channels)]
    
    # Apply window
    noise_time_windowed = [noise_time[k] * window_function for k in range(N_channels)]
    
    if return_time_domain:
        return xp.asarray(noise_time_windowed)
    else:
        # Transform back to frequency domain
        noise_freq_windowed = [xp.fft.rfft(noise_time_windowed[k]) * delta_t for k in range(N_channels)]
        return xp.asarray(noise_freq_windowed)


def pad_to_length(array, N_t, pad_value=1.0, pad_mode='end', use_gpu=False):
    """
    Pad array to specified length.

    Parameters
    ----------
    array : array-like
        Array to pad
    N_t : int
        Target length
    pad_value : float, optional
        Value to use for padding. Default: 1.0
    pad_mode : str, optional
        Where to pad: 'end', 'start', or 'both'. Default: 'end'
    use_gpu : bool, optional
        Use GPU arrays. Default: False

    Returns
    -------
    array
        Padded array of length N_t
    """
    xp = np
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            pass

    array = xp.asarray(array)
    current_length = len(array)

    if current_length >= N_t:
        return array[:N_t]

    pad_length = N_t - current_length

    if pad_mode == 'end':
        padding = xp.full(pad_length, pad_value)
        return xp.concatenate([array, padding])
    elif pad_mode == 'start':
        padding = xp.full(pad_length, pad_value)
        return xp.concatenate([padding, array])
    elif pad_mode == 'both':
        pad_left = pad_length // 2
        pad_right = pad_length - pad_left
        padding_left = xp.full(pad_left, pad_value)
        padding_right = xp.full(pad_right, pad_value)
        return xp.concatenate([padding_left, array, padding_right])
    else:
        raise ValueError(f"Unknown pad_mode: {pad_mode}")
