"""
Waveform Generation for MBH Binaries

This module provides functions and classes for generating Massive Black Hole (MBH)
binary waveforms using BBHx, with support for data gaps.

Functions
---------
MBH_f : Generate MBH waveforms with M, q parametrization
get_array_module : Detect numpy vs cupy arrays

Classes
-------
GapBBHWaveformFD : Gap-aware waveform generator
"""

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from lisatools.utils.constants import PC_SI, YRSID_SI


def get_array_module(array):
    """
    Determine if array is numpy or cupy.

    Parameters
    ----------
    array : array-like
        Input array to check

    Returns
    -------
    module
        numpy or cupy module
    """
    if hasattr(array, '__cuda_array_interface__'):
        try:
            import cupy as cp
            return cp
        except ImportError:
            return np
    return np


def MBH_f(wave_gen, M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref, **kwargs):
    """
    Generate massive black hole waveforms with M, q parametrization.

    Changes parametrization from (m1, m2, dist_m) to (M, q, dist_Gpc).
    Returns TDI A, E channels for MBH binary.

    Parameters
    ----------
    wave_gen : BBHWaveformFD
        BBHx waveform generator instance
    M : float
        Total mass (solar masses)
    q : float
        Mass ratio q = m2/m1 (q <= 1)
    a1 : float
        Primary spin magnitude  
    a2 : float
        Secondary spin magnitude
    inc : float
        Inclination angle (radians)
    dist_Gpc : float
        Luminosity distance (Gpc)
    phi_ref : float
        Reference phase (radians)
    lam : float
        Ecliptic longitude (radians)
    beta : float
        Ecliptic latitude (radians)
    psi : float
        Polarization angle (radians)
    t_ref : float
        Reference time (seconds)
    **kwargs : dict
        Must contain:
        - 'freq': frequency array (Hz)
        - 'f_ref': reference frequency (Hz)
        - 'modes': list of (l,m) tuples

    Returns
    -------
    array
        Waveform in frequency domain, shape (2, N_freq) for channels A, E
        Units: Hz^-1 (BBHx standard normalization)

    Notes
    -----
    Uses BBHx's standard normalization convention (no delta_t scaling).
    Component masses are computed as:
        m1 = q * M / (1 + q)  (secondary, smaller)
        m2 = M / (1 + q)      (primary, larger)
    """
    freq = kwargs['freq']
    f_ref = kwargs['f_ref']
    modes = kwargs['modes']
    
    # Convert to component masses
    m1 = q * M / (1.0 + q)
    m2 = M / (1.0 + q)
    
    # Convert distance to meters
    dist_m = dist_Gpc * 1e9 * PC_SI
    
    # Generate waveform using BBHx
    MBH_AET = wave_gen(
        m1, m2, a1, a2, dist_m, phi_ref, f_ref, inc, lam, beta, psi, t_ref,
        freqs=freq, modes=modes, direct=False, fill=True, squeeze=True,
        length=len(freq)
    )[0][0:2]  # Return only A, E channels

    return MBH_AET


class GapBBHWaveformFD:
    """
    Waveform generator for MBHs with time-domain gap application.

    This class wraps BBHWaveformFD and applies gaps in the time domain via:
    1. Generate waveform in frequency domain
    2. IFFT to time domain
    3. Apply gap window function
    4. FFT back to frequency domain
    5. Scale by 1/delta_t to match user's convention

    The gap window is fixed throughout sampling (typically one realization per MCMC run).

    Parameters
    ----------
    wave_gen : BBHWaveformFD
        BBHx waveform generator instance
    gap_window_array : array-like
        Time-domain gap window function, shape (N_time,)
        Should be 1 where data is good, 0 in gaps, with smooth tapers
    delta_t : float
        Time-domain sampling interval (seconds)
    use_gpu : bool, optional
        Whether to use GPU acceleration. Default: False

    Attributes
    ----------
    wave_gen : BBHWaveformFD
        Underlying BBHx waveform generator
    gap_window_array : array
        Gap window in time domain
    delta_t : float
        Sampling interval
    xp : module
        numpy or cupy depending on use_gpu
    num_bin_all : int
        Number of binaries (for compatibility with BBHx interface)

    Notes
    -----
    This class maintains the same interface as BBHWaveformFD so it can be
    used as a drop-in replacement in likelihood computations.

    The waveform scaling convention matches the user's code where the output
    is divided by delta_t, so that inner products have the form:
        ⟨a|b⟩ = 4·Δt·Re[Σ ã(f)·b̃*(f) / (N·Sn(f))]
    """

    def __init__(self, wave_gen, gap_window_array, delta_t, use_gpu=False):
        """
        Initialize gap-aware waveform generator.

        Parameters
        ----------
        wave_gen : BBHWaveformFD
            BBHx waveform generator
        gap_window_array : array-like
            Time-domain gap window (length N_time)
        delta_t : float
            Time sampling interval (seconds)
        use_gpu : bool
            Use GPU acceleration
        """
        self.wave_gen = wave_gen
        self.delta_t = delta_t
        self.use_gpu = use_gpu

        # Select array backend
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
                self.gap_window_array = cp.asarray(gap_window_array)
            except ImportError:
                print("Warning: CuPy not available, falling back to NumPy")
                self.xp = np
                self.gap_window_array = np.asarray(gap_window_array)
        else:
            self.xp = np
            self.gap_window_array = np.asarray(gap_window_array)

    def __call__(self, M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
                 freq=None, f_ref=0.0, modes=None, N_channels=2, **kwargs):
        """
        Generate gap-masked MBH waveform.

        This method follows the same parametrization as MBH_f function:
        - Total mass M and mass ratio q instead of (m1, m2)
        - Distance in Gpc instead of meters
        - Returns waveform scaled by 1/delta_t

        Parameters
        ----------
        M : float
            Total mass (solar masses)
        q : float
            Mass ratio q = m2/m1 (q <= 1)
        a1 : float
            Primary spin magnitude
        a2 : float
            Secondary spin magnitude
        inc : float
            Inclination angle (radians)
        dist_Gpc : float
            Luminosity distance (Gpc)
        phi_ref : float
            Reference phase (radians)
        lam : float
            Ecliptic longitude (radians)
        beta : float
            Ecliptic latitude (radians)
        psi : float
            Polarization angle (radians)
        t_ref : float
            Reference time (seconds)
        freq : array, optional
            Frequency array for evaluation (Hz)
        f_ref : float, optional
            Reference frequency (Hz). Default: 0.0 (let PhenomHM choose)
        modes : list, optional
            List of (l,m) mode tuples. Default: [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        N_channels : int, optional
            Number of TDI channels to return. Default: 2 (A, E)
        **kwargs : dict
            Additional arguments passed to BBHWaveformFD

        Returns
        -------
        array
            Gap-masked waveform in frequency domain, shape (N_channels, N_freq)
            Scaled by 1/delta_t to match user's convention

        Notes
        -----
        The function performs:
        1. Convert (M, q, dist_Gpc) → (m1, m2, dist_m) for BBHx
        2. Generate waveform using BBHx
        3. IFFT → apply gap → FFT
        4. Scale by 1/delta_t
        """

        # Convert user's parametrization to BBHx parametrization
        m1 = q * M / (1.0 + q)  # Secondary mass (smaller)
        m2 = M / (1.0 + q)       # Primary mass (larger)
        dist_m = dist_Gpc * 1e9 * PC_SI  # Gpc → meters

        # Set default modes if not provided
        if modes is None:
            modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]

        # Generate waveform using BBHx (returns TDI A, E, T)
        waveform_f_AET = self.wave_gen(
            m1, m2, a1, a2, dist_m, phi_ref, f_ref, inc, lam, beta, psi, t_ref,
            freqs=freq, modes=modes, direct=False, fill=True, squeeze=True,
            length=len(freq), **kwargs
        )[0]

        # Extract only the channels we need (typically A, E)
        waveform_f = waveform_f_AET[0:N_channels]

        # Apply gap in time domain (following user's approach)
        # Step 1: IFFT to time domain
        # BBHx waveforms have units Hz^-1, must divide by delta_t before IRFFT
        waveform_t = self.xp.asarray([
            self.xp.fft.irfft(waveform_f[k] / self.delta_t)
            for k in range(N_channels)
        ])

        # Step 2: Apply gap window
        waveform_t_gapped = waveform_t * self.gap_window_array

        # Step 3: FFT back to frequency domain
        # Need to multiply by delta_t to get back to units Hz^-1
        waveform_f_gapped = self.xp.asarray([
            self.xp.fft.rfft(waveform_t_gapped[k]) * self.delta_t
            for k in range(N_channels)
        ])

        # Return waveform in frequency domain with BBHx normalization (Hz^-1)
        return waveform_f_gapped

    @property
    def num_bin_all(self):
        """Number of binaries (for BBHx interface compatibility)."""
        return 1
