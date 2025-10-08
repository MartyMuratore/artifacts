import scipy
from lisatools.sensitivity import (
    get_sensitivity,
    A1TDISens,
    E1TDISens,
    T1TDISens,
    LISASens,
)
from lisatools.utils.constants import lisaLT

import numpy as np
import os


def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product using BBHx's standard normalization.

    Uses: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]
    """
    return 4 * delta_f * xp.real(xp.sum(signal_1_f * signal_2_f.conj() / PSD))


def sensitivity_LWA(f):
    """
    LISA sensitivity function in the long-wavelength approximation (https://arxiv.org/pdf/1803.01944.pdf).

    args:
        f (float): LISA-band frequency of the signal

    Returns:
        The output sensitivity strain Sn(f)
    """

    # Defining supporting functions
    L = 2.5e9  # m
    fstar = 19.09e-3  # Hz

    P_OMS = (1.5e-11**2) * (1 + (2e-3 / f) ** 4)  # Hz-1
    P_acc = (3e-15**2) * (1 + (0.4e-3 / f) ** 2) * (1 + (f / 8e-3) ** 4)  # Hz-1

    # S_c changes depending on signal duration (Equation 14 in 1803.01944)
    # for 1 year
    alpha = 0.171
    beta = 292
    kappa = 1020
    gamma = 1680
    fk = 0.00215
    # log10_Sc = (np.log10(9)-45) -7/3*np.log10(f) -(f*alpha + beta*f*np.sin(kappa*f))*np.log10(np.e) + np.log10(1 + np.tanh(gamma*(fk-f))) #Hz-1

    A = 9e-45
    Sc = (
        A
        * f ** (-7 / 3)
        * np.exp(-(f**alpha) + beta * f * np.sin(kappa * f))
        * (1 + np.tanh(gamma * (fk - f)))
    )
    sensitivity = (10 / (3 * L**2)) * (
        P_OMS + 4 * (P_acc) / ((2 * np.pi * f) ** 4)
    ) * (1 + 6 * f**2 / (10 * fstar**2)) + Sc
    return sensitivity


def write_psd_file(
    model="scirdv1",
    channels="AET",
    tdi2=True,
    include_foreground=False,
    filename="example_psd.npy",
    **kwargs,
):
    """
    Write a PSD file for a given model.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'. if None, the sensitivity curve without projections is computed.
    tdi2 : bool
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.
    """

    assert channels in [
        None,
        "A",
        "AE",
        "AET",
    ], "channels must be None, 'A', 'AE', or 'AET'"
    if include_foreground:
        assert (
            "stochastic_params" in kwargs.keys()
        ), "`stochastic_params = List(Tobs) [s]` must be provided if include_foreground is True"

    freqs = np.linspace(0, 1, 100001)[1:]

    if channels is None:
        sens_fns = [LISASens]

        default_kwargs = dict(return_type="PSD", average=False)

    elif "A" in channels:
        sens_fns = [A1TDISens]
        if "E" in channels:
            sens_fns.append(E1TDISens)
        if "T" in channels:
            sens_fns.append(T1TDISens)

        default_kwargs = dict(
            return_type="PSD",
        )

    updated_kwargs = default_kwargs | kwargs

    Sn = [
        get_sensitivity(freqs, sens_fn=sens_fn, model=model, **updated_kwargs)
        for sens_fn in sens_fns
    ]

    if tdi2:
        x = 2.0 * np.pi * lisaLT * freqs
        tdi_factor = 4 * np.sin(2 * x) ** 2
        Sn = [sens * tdi_factor for sens in Sn]

    Sn = np.array(Sn)
    np.save(filename, np.vstack((freqs, Sn)).T)


def load_psd_from_file(psd_file, xp=np):
    """
    Load the PSD from a file and return an interpolant.

    Parameters
    ----------
    psd_file : str
        The name of the file to load the PSD from.
    xp : module
        The module to use for array operations. Default is np.

    Returns
    -------
    psd_clipped : function
        A function that takes a frequency and returns the PSD at that frequency.
    """

    psd_in = np.load(psd_file).T
    freqs, values = psd_in[0], np.atleast_2d(psd_in[1:])

    # convert to cupy if needed
    freqs = xp.asarray(freqs)
    values = xp.asarray(values)

    backend = "cpu" if xp is np else "gpu"
    print(f"Using {backend} backend for PSD interpolation")

    min_psd = np.min(
        values[:, freqs < 1e-2], axis=1
    )  # compatible with both tdi 1 and tdi 2
    max_psd = np.max(values, axis=1)
    psd_interp = scipy.interpolate.CubicSpline(freqs, values, axis=1, extrapolate=True)

    def psd_clipped(f, **kwargs):
        f = xp.clip(f, 0.00001, 1.0)

        out = xp.array(
            [
                xp.clip(xp.atleast_2d(psd_interp(f))[i], min_psd[i], max_psd[i])
                for i in range(len(values))
            ]
        )
        return xp.squeeze(
            out
        )  # remove the extra dimension if there is only one channel

    return psd_clipped


def load_psd(
    logger,
    model="scirdv1",
    channels="AET",
    tdi2=True,
    include_foreground=False,
    filename="example_psd.npy",
    xp=np,
    **kwargs,
):
    """
    Load the PSD from a file and returns an interpolant. If the file does not exist, it will be created.

    Parameters
    ----------
    model : str
        The noise model to use. Default is 'scirdv1'.
    channels : str
        The channels to include in the PSD. Default is 'AET'.
    tdi2 : bool
        Whether to use Second generation TDI. Default is True.
    include_foreground : bool
        Whether to include the foreground noise. Default is False. This is just an extra check, the actual
        argument is in the kwargs.
    filename : str
        The name of the file to save the PSD to. Default is 'example_psd.npy'.
    xp : module
        The module to use for array operations. Default is np.
    **kwargs : dict
        Additional keyword arguments to pass to the PSD generation function.

    Returns
    -------
    psd_clipped : function
        A function that takes a frequency and returns the PSD at that frequency
    """
    if filename is None or filename == "None":
        tdi_gen = "tdi2" if tdi2 else "tdi1"
        foreground = "wd" if include_foreground else "no_wd"
        filename = f"noise_psd_{model}_{channels}_{tdi_gen}_{foreground}.npy"
    if not os.path.exists(filename):
        logger.warning(f"PSD file {filename} does not exist. Creating it now.")
        write_psd_file(
            model=model,
            channels=channels,
            tdi2=tdi2,
            include_foreground=include_foreground,
            filename=filename,
            **kwargs,
        )

    logger.info(f"Loading PSD from {filename}")
    return load_psd_from_file(filename, xp=xp)


# Handle array backend selection
def get_array_module(array, CUPY_AVAILABLE=False):
    """Get the appropriate array module (numpy or cupy) for the given array."""
    if CUPY_AVAILABLE and hasattr(array, '__cuda_array_interface__'):
        return cp
    return np

def generate_colored_noise(variance_noise_AET, delta_t, seed=0, window_function=None, return_time_domain=False):
    """
    Generate colored noise with specified noise variance per frequency bin.
    
    Parameters:
    -----------
    variance_noise_f : array-like
        Noise variance per frequency bin. Can be:
        - 1D array (n_freq,) - same variance applied to both channels
        - 2D array (n_channels, n_freq) - different variance per channel
        This is the variance of the noise at each frequency
    seed : int, optional
        Random seed for reproducible noise generation. Default: 0
    window_function : array-like, optional
        Time-domain window to apply (e.g., for gaps). If None, no windowing applied.
        Shape should match time-domain length.
    return_time_domain : bool, optional
        If True, return time-domain noise. If False, return frequency-domain. Default: False
        
    Returns:
    --------
    array : Noise realization
        Shape (2, n_freq) if frequency domain, or (2, n_time) if time domain
        Always returns 2 channels (for TDI A and E channels)
        
    Notes:
    -----
    - Generates complex Gaussian noise with proper normalization for real FFTs
    - DC and Nyquist components are real-valued as required
    - If window is applied, noise is transformed to time domain, windowed, then back to frequency
    - Always generates 2 channels matching your original function behavior
    """
    # Get appropriate array module  
    xp = get_array_module(variance_noise_AET[0])
     
    # Ensure variance is an array and determine shape
    
    N_channels = 2  # Always generate 2 channels like your original
      


    xp.random.seed(seed)
    noise_f_AET_real = [xp.random.normal(0,xp.sqrt(variance_noise_AET[k])) for k in range(N_channels)]
    noise_f_AET_imag = [xp.random.normal(0,xp.sqrt(variance_noise_AET[k])) for k in range(N_channels)]

    # Compute noise in frequency domain
    noise_f_AET = xp.asarray([noise_f_AET_real[k] + 1j * noise_f_AET_imag[k] for k in range(N_channels)])

    for i in range(N_channels):
        noise_f_AET[i][0] = noise_f_AET[i][0].real + 0j
        noise_f_AET[i][-1] = noise_f_AET[i][-1].real + 0j 

    
    # Apply window function if provided
    if window_function is not None:
        return _apply_window_to_noise(noise_f_AET, delta_t, window_function, xp, return_time_domain)
    
    # Return based on requested domain
    if return_time_domain:
        # Convert to time domain
        noise_time = xp.array([xp.fft.irfft(noise_f_AET[ch])/delta_t for ch in range(n_channels)])
        return noise_time
    else:
        return noise_f_AET


def _apply_window_to_noise(noise_freq, delta_t, window_function, xp, return_time_domain):
    """
    Apply time-domain window to frequency-domain noise.
    
    Parameters:
    -----------
    noise_freq : array
        Frequency-domain noise, shape (n_channels, n_freq)
    window_function : array-like
        Time-domain window function
    xp : module
        Array module (numpy or cupy)
        
    Returns:
    --------
    array : Windowed noise in frequency domain, shape (n_channels, n_freq)
    """
    window_function = xp.asarray(window_function)
    n_channels = noise_freq.shape[0]
    
    windowed_noise = []
    
    for channel in range(n_channels):
        # Transform to time domain
        noise_time = (1/delta_t) * xp.fft.irfft(noise_freq[channel], n = len(window_function))
         
        # Apply window
        windowed_time = window_function * noise_time
        
        # Transform back to frequency domain
        windowed_freq = delta_t *xp.fft.rfft(windowed_time)
        windowed_noise.append(windowed_freq)

    if return_time_domain:
        return xp.asarray([(1/delta_t) * xp.fft.irfft(windowed_noise[k]) for k in range(n_channels)])
    else: 
        return xp.asarray(windowed_noise)

def pad_to_length(array, N_t, pad_value=1.0, pad_mode='end', use_gpu=False):
    """
    Pad a gap_window_generator array (or any array) to a desired length N_t.
    
    This function is GPU and CPU agnostic, handling both numpy and cupy arrays
    depending on the use_gpu flag and availability of cupy.
    
    Parameters
    ----------
    array : numpy.ndarray or cupy.ndarray
        Input array to be padded. Can be 1D or 2D.
        For 2D arrays, padding is applied along the last axis (time axis).
    N_t : int
        Target length for the array after padding.
    pad_value : float, optional
        Value to use for padding. Default is 0.0.
        For gap masks, you might want to use 1.0 (valid data) or 0.0 (gap).
    pad_mode : str, optional
        Where to add padding. Options:
        - 'end': Add padding at the end (default)
        - 'start': Add padding at the beginning  
        - 'both': Add padding symmetrically on both sides
    use_gpu : bool, optional
        Whether to use GPU acceleration with cupy. Default is False.
        If True but cupy is not available, falls back to numpy with a warning.
        
    Returns
    -------
    numpy.ndarray or cupy.ndarray
        Padded array of length N_t (or shape [..., N_t] for 2D arrays).
        Returns same type as input (numpy/cupy) unless use_gpu is specified.
        
    Examples
    --------
    >>> import numpy as np
    >>> from gap_window_generator import GapWindowGenerator
    >>> 
    >>> # Create a simple gap mask
    >>> mask = np.ones(100)
    >>> mask[40:60] = 0  # Add a gap
    >>> 
    >>> # Pad to desired length
    >>> padded_mask = pad_to_length(mask, N_t=128, pad_value=1.0)
    >>> print(f"Original length: {len(mask)}, Padded length: {len(padded_mask)}")
    >>> 
    >>> # For 2D arrays (multiple channels)
    >>> mask_2d = np.ones((2, 100))
    >>> mask_2d[:, 40:60] = 0
    >>> padded_mask_2d = pad_to_length(mask_2d, N_t=128)
    >>> print(f"Original shape: {mask_2d.shape}, Padded shape: {padded_mask_2d.shape}")
    >>> 
    >>> # GPU usage (if available)
    >>> padded_mask_gpu = pad_to_length(mask, N_t=128, use_gpu=True)
    
    Raises
    ------
    ValueError
        If N_t is smaller than the current array length and truncation would be needed.
    ValueError
        If pad_mode is not one of the supported options.
    """
    
    # Determine array library to use
    if use_gpu and CUPY_AVAILABLE:
        xp = cp
        # Convert input to cupy if it's numpy
        if isinstance(array, np.ndarray):
            array = cp.asarray(array)
    else:
        xp = np
        if use_gpu and not CUPY_AVAILABLE:
            print("Warning: cupy not available, falling back to numpy")
        # Convert input to numpy if it's cupy
        if CUPY_AVAILABLE and isinstance(array, cp.ndarray):
            array = array.get()
    
    # Validate inputs
    if not isinstance(N_t, int) or N_t <= 0:
        raise ValueError("N_t must be a positive integer")
    
    if pad_mode not in ['end', 'start', 'both']:
        raise ValueError("pad_mode must be one of: 'end', 'start', 'both'")
    
    # Handle 1D and 2D arrays
    array = xp.atleast_1d(array)
    
    if array.ndim == 1:
        current_length = len(array)
    elif array.ndim == 2:
        current_length = array.shape[1]  # Assume time is last axis
    else:
        raise ValueError("Only 1D and 2D arrays are supported")
    
    # Check if padding is needed
    if current_length == N_t:
        return array
    elif current_length > N_t:
        raise ValueError(f"Array length {current_length} is greater than target N_t={N_t}. "
                        f"Truncation not supported. Consider using array[:N_t] if truncation is desired.")
    
    # Calculate padding needed
    pad_needed = N_t - current_length
    
    # Determine padding distribution
    if pad_mode == 'end':
        pad_left = 0
        pad_right = pad_needed
    elif pad_mode == 'start':
        pad_left = pad_needed
        pad_right = 0
    elif pad_mode == 'both':
        pad_left = pad_needed // 2
        pad_right = pad_needed - pad_left
    
    # Create padding arrays
    if array.ndim == 1:
        left_pad = xp.full(pad_left, pad_value, dtype=array.dtype)
        right_pad = xp.full(pad_right, pad_value, dtype=array.dtype)
        padded_array = xp.concatenate([left_pad, array, right_pad])
    else:  # 2D array
        n_channels = array.shape[0]
        left_pad = xp.full((n_channels, pad_left), pad_value, dtype=array.dtype)
        right_pad = xp.full((n_channels, pad_right), pad_value, dtype=array.dtype)
        padded_array = xp.concatenate([left_pad, array, right_pad], axis=1)
    
    return padded_array