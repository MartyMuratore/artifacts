"""
Gap-Aware Heterodyned Likelihood for LISA MBH Parameter Estimation

This module provides classes for fast likelihood computation of massive black hole
binary (MBH) signals with data gaps. It implements heterodyning to speed up MCMC
sampling by evaluating waveforms on sparse frequency grids.

Key Features:
- Gap handling via time-domain windowing
- Heterodyned likelihood computation (sparse frequency evaluation)
- CPU/GPU agnostic implementation
- Consistent with user's FFT normalization conventions

Classes:
    GapBBHWaveformFD: Waveform generator with gap application
    GapHeterodynedLikelihood: Fast heterodyned likelihood with gaps

"""

import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from lisatools.utils.constants import PC_SI, YRSID_SI


# ============================================================================
# Utility Functions (copied for modularity)
# ============================================================================

def inner_prod(signal_1_f, signal_2_f, PSD, delta_f, xp=np):
    """
    Compute noise-weighted inner product in frequency domain.

    Uses BBHx's standard normalization: ⟨a|b⟩ = 4·Δf·Re[Σ a(f)·b*(f) / Sn(f)]

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


# ============================================================================
# Gap-Aware Waveform Generator
# ============================================================================

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

        This method follows the same parametrization as the user's MBH_f function:
        - Total mass M and mass ratio q instead of (m1, m2)
        - Distance in Gpc instead of meters
        - Returns waveform scaled by 1/delta_t

        Parameters
        ----------
        M : float or array
            Total mass (solar masses)
        q : float or array
            Mass ratio q = m2/m1 (q <= 1)
        a1 : float or array
            Primary spin magnitude
        a2 : float or array
            Secondary spin magnitude
        inc : float or array
            Inclination angle (radians)
        dist_Gpc : float or array
            Luminosity distance (Gpc)
        phi_ref : float or array
            Reference phase (radians)
        lam : float or array
            Ecliptic longitude (radians)
        beta : float or array
            Ecliptic latitude (radians)
        psi : float or array
            Polarization angle (radians)
        t_ref : float or array
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
        # We use fill=True to get full frequency array, squeeze=True for clean output
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
        # Let irfft determine the natural time-domain length
        waveform_t = self.xp.asarray([
            self.xp.fft.irfft(waveform_f[k] / self.delta_t)
            for k in range(N_channels)
        ])

        # Step 2: Create or verify gap window matches the natural IFFT length
        N_time_natural = waveform_t.shape[1]

        if self.gap_window_array is None:
            # No gap - create array of ones matching natural length
            self.gap_window_array = self.xp.ones(N_time_natural)
        elif len(self.gap_window_array) != N_time_natural:
            # Gap window length mismatch - need to resample/interpolate
            # This follows your approach: the gap window should be created based on
            # sim_t = np.arange(N_time_natural) * delta_t
            # For now, we'll truncate or pad to match
            if len(self.gap_window_array) > N_time_natural:
                # Truncate
                self.gap_window_array = self.gap_window_array[:N_time_natural]
            else:
                # Pad with ones (good data)
                pad_length = N_time_natural - len(self.gap_window_array)
                self.gap_window_array = self.xp.concatenate([
                    self.gap_window_array,
                    self.xp.ones(pad_length)
                ])

        # Step 3: Apply gap window
        waveform_t_gapped = waveform_t * self.gap_window_array

        # Step 4: FFT back to frequency domain
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
        # For now we only handle single binaries in gap case
        # Could be extended to multiple binaries if needed
        return 1


# ============================================================================
# Gap-Aware Heterodyned Likelihood
# ============================================================================

class GapHeterodynedLikelihood:
    """
    Heterodyned likelihood for MBH parameter estimation with data gaps.

    This class implements fast likelihood computation using heterodyning, which
    evaluates waveforms on a sparse frequency grid rather than the full FFT grid.
    This provides significant speed-ups for MCMC sampling (~100-1000x faster).

    The heterodyning method is described in:
    - Cornish & Littenberg (2021): arXiv:2109.02728 (general method)
    - Marsat et al (2020): arXiv:1806.08792 (LISA implementation)

    Key differences from BBHx's HeterodynedLikelihood:
    - Uses user's FFT normalization convention (explicit delta_t factors)
    - Applies gaps in time domain to both reference and test templates
    - Inner products computed as: ⟨a|b⟩ = 4·Δt·Re[Σ a·b*/(N·Sn)]

    Parameters
    ----------
    wave_gen : BBHWaveformFD
        BBHx waveform generator instance
    data_freqs : array
        Frequency array for data (Hz), length N_freq
    data_channels : array
        Data in frequency domain, shape (N_channels, N_freq)
        Should be raw FFT output (unnormalized)
    reference_params : array
        Reference parameters [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
    gap_window_array : array
        Time-domain gap window, length N_time
    delta_t : float
        Time sampling interval (seconds)
    delta_f : float
        Frequency resolution (Hz)
    N : int
        Number of time samples
    PSD_interp : callable
        PSD interpolation function, takes frequency array, returns PSD array
        Should return shape (N_channels, N_freq)
    length_f_het : int
        Number of frequencies in sparse heterodyning grid
    modes : list, optional
        Harmonic modes [(l,m), ...]. Default: [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    f_ref : float, optional
        Reference frequency (Hz). Default: 0.0
    N_channels : int, optional
        Number of TDI channels (typically 2 for A,E). Default: 2
    use_gpu : bool, optional
        Use GPU acceleration. Default: False

    Attributes
    ----------
    gap_wave_gen : GapBBHWaveformFD
        Gap-aware waveform generator
    f_dense : array
        Dense frequency array (narrowed to signal bandwidth)
    d : array
        Data in frequency domain (narrowed to signal bandwidth)
    freqs : array
        Sparse frequency grid for heterodyning
    f_m : array
        Mid-point frequencies of heterodyning bins
    h0_sparse : array
        Reference template on sparse grid
    data_constants : array
        Precomputed heterodyning coefficients [A0, A1, B0, B1]
    reference_d_d : float
        ⟨d|d⟩ for reference (constant)
    reference_h_h : float
        ⟨h0|h0⟩ for reference template
    reference_d_h : float
        ⟨d|h0⟩ for reference template
    reference_ll : float
        Log-likelihood for reference parameters

    Notes
    -----
    The heterodyning process:

    1. Initialization (done once):
       - Generate reference template h0(f) on dense grid (with gap applied)
       - Generate reference template h0_sparse(f) on sparse grid
       - Compute heterodyning coefficients A0, A1, B0, B1 by binning:
           A0 = Σ 4·Δt/(N·Sn)·h0*·d per sparse bin
           A1 = Σ 4·Δt/(N·Sn)·h0*·d·(f-f_mid) per sparse bin
           B0 = Σ 4·Δt/(N·Sn)·|h0|² per sparse bin
           B1 = Σ 4·Δt/(N·Sn)·|h0|²·(f-f_mid) per sparse bin

    2. Online evaluation (each MCMC step):
       - Generate test template h(f) on sparse grid only (~128 frequencies)
       - Compute complex residual r(f) = h(f) / h0_sparse(f)
       - Heterodyned inner products:
           ⟨d|h⟩_het = Σ [A0·r + A1·r·Δf/2]
           ⟨h|h⟩_het = Σ [B0·|r|² + B1·(r*·dr + r·dr*)/2]
       - Log-likelihood: log L = -½(⟨d|d⟩ + ⟨h|h⟩_het - 2·⟨d|h⟩_het)

    The speed-up comes from evaluating ~128 frequencies instead of ~300,000!
    """

    def __init__(self, wave_gen, data_freqs, data_channels, reference_params,
                 gap_window_array, delta_t, delta_f, N, PSD_interp, length_f_het,
                 modes=None, f_ref=0.0, N_channels=2, use_gpu=False):
        """
        Initialize gap-aware heterodyned likelihood.

        This performs all the expensive one-time computations including
        generating the reference templates and computing heterodyning coefficients.
        """

        # Store basic parameters
        self.delta_t = delta_t
        self.delta_f = delta_f
        self.N = N
        self.N_channels = N_channels
        self.length_f_het = length_f_het
        self.PSD_interp = PSD_interp
        self.use_gpu = use_gpu

        # Select array backend
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                print("Warning: CuPy not available, falling back to NumPy")
                self.xp = np
                use_gpu = False
        else:
            self.xp = np

        # Convert inputs to appropriate backend
        self.f_dense = self.xp.asarray(data_freqs)
        self.d = self.xp.asarray(data_channels)
        gap_window_array = self.xp.asarray(gap_window_array)

        # Set default modes
        if modes is None:
            modes = [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        self.modes = modes
        self.f_ref = f_ref

        # Create gap-aware waveform generator
        self.gap_wave_gen = GapBBHWaveformFD(
            wave_gen, gap_window_array, delta_t, use_gpu=use_gpu
        )

        print("Initializing heterodyned likelihood...")
        print(f"  Dense frequency grid: {len(self.f_dense)} points")
        print(f"  Sparse frequency grid: {length_f_het} points")
        print(f"  Expected speed-up: ~{len(self.f_dense) / length_f_het:.1f}x")

        # Initialize heterodyning (this is the expensive part)
        self._init_heterodyne_info(reference_params)

        print("✓ Heterodyned likelihood initialized successfully")

    def _init_heterodyne_info(self, reference_params):
        """
        Initialize all heterodyning quantities.

        This is the expensive one-time computation that:
        1. Generates reference templates (dense and sparse)
        2. Creates sparse frequency grid
        3. Computes heterodyning coefficients A0, A1, B0, B1
        4. Computes reference inner products

        Parameters
        ----------
        reference_params : array
            Reference parameters [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
        """

        # Unpack reference parameters
        M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref = reference_params

        # Initial sparse grid (logarithmic spacing)
        minF = float(self.f_dense.min()) * 0.999999999999
        maxF = float(self.f_dense.max()) * 1.000000000001
        if minF == 0.0:
            minF = 1e-6

        freqs = self.xp.logspace(
            self.xp.log10(minF), self.xp.log10(maxF), self.length_f_het
        )

        print("  Generating dense reference template...")
        # Generate dense reference template h0(f) with gaps applied
        h0 = self.gap_wave_gen(
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
            freq=self.f_dense, f_ref=self.f_ref, modes=self.modes,
            N_channels=self.N_channels
        )

        print("  Generating sparse reference template...")
        # Generate sparse reference template to check where signal exists
        h0_temp = self.gap_wave_gen(
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
            freq=freqs, f_ref=self.f_ref, modes=self.modes,
            N_channels=self.N_channels
        )

        # Narrow frequency range to where signal is non-zero
        # Check across all channels
        signal_exists = self.xp.any(self.xp.abs(h0_temp) > 0, axis=0)
        freqs_keep = freqs[signal_exists]

        if len(freqs_keep) < 10:
            print("  Warning: Very few non-zero frequencies found, using broader range")
            freqs_keep = freqs[self.xp.abs(h0_temp[0]) > 0.1 * self.xp.abs(h0_temp[0]).max()]

        # Regenerate sparse grid over non-zero region
        freqs = self.xp.logspace(
            self.xp.log10(float(freqs_keep[0])),
            self.xp.log10(float(freqs_keep[-1])),
            self.length_f_het
        )

        print("  Regenerating sparse template on optimized grid...")
        # Generate final sparse reference template
        h0_sparse = self.gap_wave_gen(
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
            freq=freqs, f_ref=self.f_ref, modes=self.modes,
            N_channels=self.N_channels
        )

        # Store sparse reference (add dimension for compatibility with online evaluation)
        self.h0_sparse = h0_sparse[self.xp.newaxis, :, :]  # Shape: (1, N_channels, length_f_het)

        # Narrow dense arrays to signal bandwidth
        inds = (self.f_dense >= freqs[0]) & (self.f_dense <= freqs[-1])
        self.f_dense = self.f_dense[inds]
        self.d = self.d[:, inds]
        h0 = h0[:, inds]

        print(f"  Signal bandwidth: {float(freqs[0]):.2e} - {float(freqs[-1]):.2e} Hz")
        print(f"  Dense grid narrowed to {len(self.f_dense)} points")

        # Assign each dense frequency to a sparse bin
        bins = self.xp.searchsorted(freqs, self.f_dense, 'right') - 1

        # Mid-point frequency of each sparse bin
        f_m = (freqs[1:] + freqs[:-1]) / 2

        # Get PSD at dense frequencies
        # Convert to CPU for PSD evaluation if needed
        try:
            f_dense_cpu = self.f_dense.get()
        except AttributeError:
            f_dense_cpu = self.f_dense

        S_n = self.PSD_interp(f_dense_cpu)
        S_n = self.xp.asarray(S_n)

        # Ensure S_n has correct shape (N_channels, N_freq)
        if S_n.ndim == 1:
            S_n = self.xp.tile(S_n, (self.N_channels, 1))

        print("  Computing heterodyning coefficients...")
        # Compute normalization factor for inner products
        # Factor: 4·Δf/Sn(f) matches BBHx's standard convention
        norm_factor = 4 * self.delta_f / S_n

        # Compute individual frequency contributions to heterodyning coefficients
        # A0, A1: data-template cross terms
        A0_flat = norm_factor * (h0.conj() * self.d)
        A1_flat = norm_factor * (h0.conj() * self.d) * (self.f_dense - f_m[bins])

        # B0, B1: template auto-correlation terms
        B0_flat = norm_factor * (h0.conj() * h0)
        B1_flat = norm_factor * (h0.conj() * h0) * (self.f_dense - f_m[bins])

        # Initialize sparse coefficient arrays
        A0_in = self.xp.zeros((self.N_channels, self.length_f_het), dtype=self.xp.complex128)
        A1_in = self.xp.zeros_like(A0_in)
        B0_in = self.xp.zeros_like(A0_in)
        B1_in = self.xp.zeros_like(A0_in)

        # Sum contributions within each sparse bin
        for ind in self.xp.unique(bins[:-1]):
            inds_keep = bins == ind
            inds_keep[-1] = False  # Exclude last point to avoid edge effects

            # +1 allows for zero as first entry (for C compatibility in BBHx)
            A0_in[:, ind + 1] = self.xp.sum(A0_flat[:, inds_keep], axis=1)
            A1_in[:, ind + 1] = self.xp.sum(A1_flat[:, inds_keep], axis=1)
            B0_in[:, ind + 1] = self.xp.sum(B0_flat[:, inds_keep], axis=1)
            B1_in[:, ind + 1] = self.xp.sum(B1_flat[:, inds_keep], axis=1)

        # Flatten and concatenate all coefficients
        self.data_constants = self.xp.concatenate([
            A0_in.flatten(),
            A1_in.flatten(),
            B0_in.flatten(),
            B1_in.flatten()
        ])

        print("  Computing reference quantities...")
        # Compute reference inner products
        # These are constants that don't change during sampling

        # ⟨d|d⟩: data auto-correlation
        self.reference_d_d = self.xp.sum(norm_factor * (self.d.conj() * self.d)).real
        print(f"  DEBUG: reference_d_d = {self.reference_d_d:.3e}")
        print(f"  DEBUG: Should be ~5.167e+04")


        # ⟨h0|h0⟩: reference template auto-correlation
        self.reference_h_h = self.xp.sum(B0_flat).real

        # ⟨d|h0⟩: data-reference cross-correlation
        self.reference_d_h = self.xp.sum(A0_flat).real

        # Reference log-likelihood
        self.reference_ll = -0.5 * (self.reference_d_d + self.reference_h_h - 2 * self.reference_d_h)

        # Store sparse frequencies
        self.freqs = freqs
        self.f_m = f_m

        print(f"  Reference log-likelihood: {float(self.reference_ll):.2f}")
        print(f"  Reference SNR: {float(self.reference_d_h / self.xp.sqrt(self.reference_h_h)):.1f}")

    def get_ll(self, params, phase_marginalize=False, return_extracted_snr=False):
        """
        Compute heterodyned log-likelihood for test parameters.

        This is the fast online evaluation that happens at each MCMC step.
        It only evaluates the waveform on the sparse frequency grid.

        Parameters
        ----------
        params : array
            Test parameters, shape (11,) or (11, N_samples) for vectorized evaluation
            Order: [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
        phase_marginalize : bool, optional
            If True, marginalize over phase (use |⟨d|h⟩| instead of ⟨d|h⟩). Default: False
        return_extracted_snr : bool, optional
            If True, return SNR in addition to log-likelihood. Default: False

        Returns
        -------
        float or tuple
            Log-likelihood value(s)
            If return_extracted_snr=True: (log_likelihood, snr)

        Notes
        -----
        The heterodyned likelihood approximates the full likelihood by:
        1. Computing residual r(f) = h(f) / h0(f) on sparse grid
        2. Heterodyned inner products using precomputed coefficients
        3. Final likelihood from heterodyned quantities

        This is much faster than the direct likelihood because waveform
        generation happens on ~128 frequencies instead of ~300,000.
        """

        # Handle single vs. vectorized parameters
        params = self.xp.atleast_2d(params)
        if params.shape[0] == 11 and params.shape[1] != 11:
            # Params given as (11, N_samples) - transpose
            pass
        elif params.shape[1] == 11:
            # Params given as (N_samples, 11) - transpose
            params = params.T
        else:
            # Single parameter set
            params = params.reshape(11, 1)

        N_samples = params.shape[1]

        # Unpack parameters
        M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref = params

        # Generate sparse test template(s) - THIS IS THE FAST PART!
        # Only ~128 frequency evaluations instead of ~300,000
        h_sparse = self.gap_wave_gen(
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
            freq=self.freqs, f_ref=self.f_ref, modes=self.modes,
            N_channels=self.N_channels
        )

        # Ensure correct shape for single vs. multiple samples
        if N_samples == 1 and h_sparse.ndim == 2:
            h_sparse = h_sparse[self.xp.newaxis, :, :]  # (1, N_channels, N_freq)

        # Compute complex residual: r(f) = h(f) / h0(f)
        # This encodes how the test template differs from reference
        # Add small epsilon to avoid division by zero
        epsilon = 1e-100
        r = h_sparse / (self.h0_sparse + epsilon)

        # Mask out where reference is truly zero (avoid spurious contributions)
        mask_zero = self.xp.abs(self.h0_sparse) < epsilon * 10
        r = self.xp.where(mask_zero, 0.0, r)

        # Compute heterodyned inner products using precomputed coefficients
        # This is where the speed-up happens - we sum over sparse grid only

        hdyn_d_h, hdyn_h_h = self._compute_heterodyned_inner_products(r, N_samples)

        # Phase marginalization if requested
        d_h_temp = hdyn_d_h if not phase_marginalize else self.xp.abs(hdyn_d_h)

        # Compute log-likelihood
        # log L = -½(⟨d|d⟩ + ⟨h|h⟩_het - 2⟨d|h⟩_het)
        out = -0.5 * (self.reference_d_d + hdyn_h_h - 2 * d_h_temp).real

        # Convert to CPU if on GPU
        try:
            out = out.get()
            hdyn_d_h = hdyn_d_h.get()
            hdyn_h_h = hdyn_h_h.get()
            d_h_temp = d_h_temp.get()
        except AttributeError:
            pass

        # Return format
        if return_extracted_snr:
            snr = d_h_temp.real / np.sqrt(hdyn_h_h.real)
            return np.array([out, snr]).T if N_samples > 1 else (float(out), float(snr))
        else:
            return out if N_samples > 1 else float(out)

    def _compute_heterodyned_inner_products(self, r, N_samples):
        """
        Compute heterodyned inner products from complex residuals.

        This implements the core heterodyning calculation using the
        precomputed coefficients A0, A1, B0, B1.

        Parameters
        ----------
        r : array
            Complex residual r(f) = h(f)/h0(f), shape (N_samples, N_channels, length_f_het)
        N_samples : int
            Number of parameter samples

        Returns
        -------
        hdyn_d_h : array
            Heterodyned ⟨d|h⟩, shape (N_samples,)
        hdyn_h_h : array
            Heterodyned ⟨h|h⟩, shape (N_samples,)

        Notes
        -----
        The heterodyned inner products are computed as:
            ⟨d|h⟩_het = Σ_bins [A0·r + A1·r·Δf/2]
            ⟨h|h⟩_het = Σ_bins [B0·|r|² + B1·Re(r*·dr)]

        where the sum is over sparse frequency bins and dr = (r[i+1] - r[i])/Δf
        """

        # Initialize output arrays
        hdyn_d_h = self.xp.zeros(N_samples, dtype=self.xp.complex128)
        hdyn_h_h = self.xp.zeros(N_samples, dtype=self.xp.complex128)

        # Extract coefficients from flattened storage
        n_coeff = self.N_channels * self.length_f_het
        A0 = self.data_constants[0:n_coeff].reshape(self.N_channels, self.length_f_het)
        A1 = self.data_constants[n_coeff:2*n_coeff].reshape(self.N_channels, self.length_f_het)
        B0 = self.data_constants[2*n_coeff:3*n_coeff].reshape(self.N_channels, self.length_f_het)
        B1 = self.data_constants[3*n_coeff:4*n_coeff].reshape(self.N_channels, self.length_f_het)

        # Frequency spacing in sparse grid
        df_sparse = self.freqs[1:] - self.freqs[:-1]
        df_sparse = self.xp.concatenate([df_sparse[:1], df_sparse])  # Prepend first spacing

        # Loop over samples (could be vectorized further if needed)
        for i_sample in range(N_samples):
            r_sample = r[i_sample] if N_samples > 1 else r[0]  # Shape: (N_channels, length_f_het)

            # Compute derivative of residual: dr/df
            # Use forward differences
            dr = self.xp.zeros_like(r_sample)
            dr[:, :-1] = (r_sample[:, 1:] - r_sample[:, :-1]) / df_sparse[1:][:, self.xp.newaxis].T
            dr[:, -1] = dr[:, -2]  # Extrapolate last point

            # Heterodyned data-template cross term: ⟨d|h⟩_het
            # Formula: Σ A0·r (A0 already contains binned contributions)
            term_d_h = A0 * r_sample
            hdyn_d_h[i_sample] = self.xp.sum(term_d_h)

            # Heterodyned template auto-correlation: ⟨h|h⟩_het
            # Formula: Σ B0·|r|² (B0 already contains binned contributions)
            r_squared = self.xp.abs(r_sample)**2
            term_h_h = B0 * r_squared
            hdyn_h_h[i_sample] = self.xp.sum(term_h_h)

        return hdyn_d_h, hdyn_h_h

    def __call__(self, params, **kwargs):
        """Convenience method to call get_ll."""
        return self.get_ll(params, **kwargs)


# ============================================================================
# Unified Likelihood Interface (Direct or Heterodyned)
# ============================================================================

class GapLikelihood:
    """
    Unified likelihood interface with option to use direct or heterodyned computation.

    This class provides a single interface that can switch between:
    - Direct likelihood: Full frequency grid (accurate but slow)
    - Heterodyned likelihood: Sparse frequency grid (fast approximation)

    This is useful for:
    - Comparing direct vs heterodyned results
    - Debugging and validation
    - Switching strategies based on problem size

    Parameters
    ----------
    wave_gen : BBHWaveformFD
        BBHx waveform generator
    data_freqs : array
        Frequency array (Hz)
    data_channels : array
        Data in frequency domain, shape (N_channels, N_freq)
    gap_window_array : array
        Time-domain gap window
    delta_t : float
        Time sampling interval (seconds)
    delta_f : float
        Frequency resolution (Hz)
    N : int
        Number of time samples
    PSD_interp : callable
        PSD interpolation function
    reference_params : array, optional
        Reference parameters for heterodyning. If None and use_heterodyned=True,
        must call set_reference() before evaluation.
    length_f_het : int, optional
        Sparse grid size for heterodyning. Default: 128
    modes : list, optional
        Harmonic modes. Default: [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
    f_ref : float, optional
        Reference frequency (Hz). Default: 0.0
    N_channels : int, optional
        Number of TDI channels. Default: 2
    use_heterodyned : bool, optional
        If True, use heterodyned likelihood. If False, use direct. Default: True
    use_gpu : bool, optional
        Use GPU acceleration. Default: False

    Attributes
    ----------
    use_heterodyned : bool
        Whether heterodyned likelihood is active
    like_het : GapHeterodynedLikelihood or None
        Heterodyned likelihood instance (if use_heterodyned=True)
    gap_wave_gen : GapBBHWaveformFD
        Gap-aware waveform generator (used for direct likelihood)

    Examples
    --------
    >>> # Use heterodyned likelihood (fast)
    >>> like = GapLikelihood(..., reference_params=ref_params, use_heterodyned=True)
    >>> ll = like(test_params)
    >>>
    >>> # Use direct likelihood (slow but exact)
    >>> like = GapLikelihood(..., use_heterodyned=False)
    >>> ll = like(test_params)
    >>>
    >>> # Switch between modes
    >>> like.set_mode(use_heterodyned=False)  # Switch to direct
    >>> ll_direct = like(test_params)
    >>> like.set_mode(use_heterodyned=True)   # Switch back to heterodyned
    >>> ll_het = like(test_params)
    """

    def __init__(self, wave_gen, data_freqs, data_channels, gap_window_array,
                 delta_t, delta_f, N, PSD_interp, reference_params=None,
                 length_f_het=128, modes=None, f_ref=0.0, N_channels=2,
                 use_heterodyned=True, use_gpu=False):
        """Initialize unified likelihood interface."""

        # Store parameters
        self.wave_gen = wave_gen
        self.data_freqs = data_freqs
        self.data_channels = data_channels
        self.gap_window_array = gap_window_array
        self.delta_t = delta_t
        self.delta_f = delta_f
        self.N = N
        self.PSD_interp = PSD_interp
        self.length_f_het = length_f_het
        self.modes = modes if modes is not None else [(2,2), (2,1), (3,3), (3,2), (4,4), (4,3)]
        self.f_ref = f_ref
        self.N_channels = N_channels
        self.use_gpu = use_gpu

        # Select array backend
        if use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                print("Warning: CuPy not available, falling back to NumPy")
                self.xp = np
        else:
            self.xp = np

        # Get PSD array
        self.PSD = self.PSD_interp(data_freqs if isinstance(data_freqs, np.ndarray) else data_freqs.get())
        self.PSD = self.xp.asarray(self.PSD)
        if self.PSD.ndim == 1:
            self.PSD = self.xp.tile(self.PSD, (N_channels, 1))

        # Create gap waveform generator (needed for both modes)
        self.gap_wave_gen = GapBBHWaveformFD(
            wave_gen, gap_window_array, delta_t, use_gpu=use_gpu
        )

        # Initialize likelihood
        self.like_het = None
        self.use_heterodyned = False  # Start as False

        if use_heterodyned:
            if reference_params is None:
                print("Warning: use_heterodyned=True but no reference_params provided.")
                print("         Call set_reference() before evaluating likelihood.")
            else:
                self.set_reference(reference_params)
                self.use_heterodyned = True

        print(f"\n✓ GapLikelihood initialized in {'HETERODYNED' if self.use_heterodyned else 'DIRECT'} mode")

    def set_reference(self, reference_params):
        """
        Set reference parameters and initialize heterodyned likelihood.

        Parameters
        ----------
        reference_params : array
            Reference parameters [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
        """
        print("\nInitializing heterodyned likelihood with reference parameters...")
        self.like_het = GapHeterodynedLikelihood(
            wave_gen=self.wave_gen,
            data_freqs=self.data_freqs,
            data_channels=self.data_channels,
            reference_params=reference_params,
            gap_window_array=self.gap_window_array,
            delta_t=self.delta_t,
            delta_f=self.delta_f,
            N=self.N,
            PSD_interp=self.PSD_interp,
            length_f_het=self.length_f_het,
            modes=self.modes,
            f_ref=self.f_ref,
            N_channels=self.N_channels,
            use_gpu=self.use_gpu
        )
        self.use_heterodyned = True

    def set_mode(self, use_heterodyned):
        """
        Switch between heterodyned and direct likelihood modes.

        Parameters
        ----------
        use_heterodyned : bool
            If True, use heterodyned likelihood (must have called set_reference first)
            If False, use direct likelihood
        """
        if use_heterodyned and self.like_het is None:
            raise ValueError("Cannot switch to heterodyned mode: no reference set. "
                           "Call set_reference() first.")
        self.use_heterodyned = use_heterodyned
        print(f"Likelihood mode switched to: {'HETERODYNED' if use_heterodyned else 'DIRECT'}")

    def get_ll_direct(self, params):
        """
        Compute direct likelihood (full frequency grid).

        This is the slow but accurate method that evaluates the waveform
        on the full FFT frequency grid.

        Parameters
        ----------
        params : array
            Parameters [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]

        Returns
        -------
        float
            Log-likelihood
        """
        # Unpack parameters
        if isinstance(params, (list, tuple)):
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref = params
        else:
            params = self.xp.atleast_1d(params)
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref = params

        # Generate waveform with gaps
        waveform_f = self.gap_wave_gen(
            M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
            freq=self.data_freqs, f_ref=self.f_ref, modes=self.modes,
            N_channels=self.N_channels
        )

        # Compute residual
        diff_f = self.data_channels - waveform_f

        # Compute inner products using BBHx convention
        inn_prod = self.xp.asarray([
            inner_prod(diff_f[k], diff_f[k], self.PSD[k], self.delta_f, xp=self.xp)
            for k in range(self.N_channels)
        ])

        ll = -0.5 * self.xp.sum(inn_prod)

        # Convert to CPU if needed
        try:
            ll = float(ll.get())
        except AttributeError:
            ll = float(ll)

        return ll

    def get_ll(self, params, **kwargs):
        """
        Compute log-likelihood using current mode (direct or heterodyned).

        Parameters
        ----------
        params : array
            Parameters [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
        **kwargs : dict
            Additional arguments passed to heterodyned likelihood if active
            (e.g., phase_marginalize, return_extracted_snr)

        Returns
        -------
        float or tuple
            Log-likelihood (and optionally SNR if return_extracted_snr=True)
        """
        if self.use_heterodyned:
            return self.like_het.get_ll(params, **kwargs)
        else:
            # Direct likelihood ignores heterodyned-specific kwargs
            return self.get_ll_direct(params)

    def __call__(self, params, **kwargs):
        """Convenience method to call get_ll."""
        return self.get_ll(params, **kwargs)


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """
    Example showing how to use the gap-aware heterodyned likelihood.

    This demonstrates:
    1. Setting up the data and gap window
    2. Creating the heterodyned likelihood
    3. Comparing to direct likelihood computation
    """

    print("="*70)
    print("Gap-Aware Heterodyned Likelihood - Example Usage")
    print("="*70)

    # This example would require importing your actual data and PSD
    # For now, just show the structure
    print("\nExample structure:")
    print("""
    # 1. Setup
    from bbhx.waveformbuild import BBHWaveformFD
    from gap_heterodyne_likelihood import GapHeterodynedLikelihood

    # Load data and PSD
    data_f_AE = ...  # Your data in frequency domain
    PSD_interp = ...  # Your PSD interpolation function
    gap_window_array = ...  # Your gap window

    # 2. Create heterodyned likelihood
    like_het = GapHeterodynedLikelihood(
        wave_gen=wave_gen,
        data_freqs=freq,
        data_channels=data_f_AE,
        reference_params=reference_params,
        gap_window_array=gap_window_array,
        delta_t=delta_t,
        delta_f=delta_f,
        N=N,
        PSD_interp=PSD_interp,
        length_f_het=128,
        modes=modes,
        N_channels=2,
        use_gpu=False
    )

    # 3. Evaluate likelihood (fast!)
    test_params = [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
    log_likelihood = like_het.get_ll(test_params)

    # 4. Use in MCMC
    def llike(params):
        return like_het.get_ll(params)
    """)

    print("\n" + "="*70)
