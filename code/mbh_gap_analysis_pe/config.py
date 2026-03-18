"""
Configuration Module for MBH Parameter Estimation

This module centralizes all paths, settings, and parameters for the MBH
parameter estimation pipeline. Modify values here to change behavior across
all analysis scripts.

Usage:
    from config import Config
    
    # Use default configuration
    cfg = Config()
    
    # Or override specific settings
    cfg = Config(use_gpu=True, use_heterodyne=True)
"""

import os
import numpy as np


class Config:
    """
    Configuration class for MBH parameter estimation pipeline.
    
    All paths, flags, and analysis parameters are defined here.
    Override any parameter by passing it to __init__.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize configuration with optional overrides.
        
        Parameters
        ----------
        **kwargs : dict
            Any configuration parameter to override (e.g., use_gpu=True)
        """
        # ==================== DIRECTORY SETTINGS ====================
        
        # Base directory for this analysis
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Data directories
        self.noise_dir = self.base_dir + "/"
        self.mcmc_dir = os.path.join(self.base_dir, "data_mcmc_simulations") + "/"
        self.fisher_dir = os.path.join(self.base_dir, "fisher_results") + "/"
        
        # External data directories (if needed)
        self.bh_directory = "/Users/ollie.burke/Documents/Work/Code/spritz_challenge/code/Light_Spritz"
        
        # Create directories if they don't exist
        os.makedirs(self.mcmc_dir, exist_ok=True)
        os.makedirs(self.fisher_dir, exist_ok=True)
        
        # ==================== COMPUTATION SETTINGS ====================
        
        # GPU/CPU settings
        self.use_gpu = False  # Set to True to use CuPy/GPU acceleration
        
        # Get appropriate array module
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                print("Warning: CuPy not available, falling back to NumPy")
                self.xp = np
                self.use_gpu = False
        else:
            self.xp = np
        
        # ==================== ANALYSIS FLAGS ====================
        
        # Plotting and debugging
        self.plot_waveform = False  # Plot waveform and noise
        self.check_snr = False      # Run SNR verification (100 noise realizations)
        
        # Gap handling mode (only one should be True)
        self.no_mask = False        # No gaps applied
        self.mask = False           # Hard mask (set gaps to NaN/zero)
        self.window = True          # Smooth windowing with Tukey tapers
        
        # Likelihood mode
        self.use_heterodyne = True  # Use heterodyned likelihood (fast, ~100-1000x speedup)
        
        # ==================== PHYSICAL/ANALYSIS PARAMETERS ====================
        
        # TDI channels to analyze
        self.n_channels = 2
        self.channels = ["A", "E"]
        self.sens_fn_calls = ["noisepsd_AE", "noisepsd_AE"]
        
        # Data files
        self.psd_filename = "tdi2_AE_w_background.npy"
        self.spritz_data_file = "LDC2_Spritz_only_GW.h5"
        
        # Time/frequency resolution
        self.delta_t = 5.0                    # Time spacing (seconds)
        self.t_obs = 1 * 31 * 24 * 3600       # Observation time (1 month, seconds)
        self.delta_f = 1 / self.t_obs         # Frequency spacing (Hz)
        
        # Frequency range
        self.f_min = 1e-5                     # Minimum frequency (Hz)
        self.f_max = None                     # Maximum frequency (Hz, None = Nyquist)
        
        # ==================== INJECTION PARAMETERS ====================
        
        # These are the "true" parameters for injection/recovery tests
        self.m1 = 1323277.47932               # Primary mass (solar masses)
        self.m2 = 612485.50602999             # Secondary mass (solar masses)
        self.a1 = 0.747377                    # Primary spin magnitude
        self.a2 = 0.8388                      # Secondary spin magnitude
        self.inc = np.pi / 3                  # Inclination (radians)
        self.dist_Gpc = 36.90249521628649     # Luminosity distance (Gpc)
        
        # Extrinsic parameters
        self.beta = -0.30300442294174235      # Ecliptic latitude (radians)
        self.lam = 1.2925183861048521         # Ecliptic longitude (radians)
        self.psi = np.pi / 6                  # Polarization angle (radians)
        
        # Reference values
        self.t_ref = 2627744.9218792617       # Reference time (seconds)
        self.f_ref = 0.0                      # Reference frequency (Hz, 0 = let code decide)
        self.phi_ref = 1.2                    # Phase at reference frequency (radians)
        
        # Derived parameters
        self.M = self.m1 + self.m2            # Total mass
        self.q = self.m2 / self.m1            # Mass ratio (q <= 1 by convention)
        
        # Waveform modes to include
        self.modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
        
        # ==================== GAP PARAMETERS ====================
        
        # Gap configuration (used if self.window = True or self.mask = True)
        self.gap_centers = [2.600e6, 0 * 2.6265e6, 2.633e6]  # Gap centers (seconds)
        self.gap_widths = [7 * 3600, 0 * 0.5 * 3600, 1 * 3600]  # Gap widths (seconds)
        self.lobe_widths = 5 * 60  # Tukey taper width (seconds, only for windowing)
        
        # lisaglitch/lisagap settings (if using those packages)
        self.include_planned = True
        self.include_unplanned = True
        self.planned_seed = None
        self.unplanned_seed = None
        
        # ==================== MCMC SETTINGS ====================
        
        # Sampler configuration
        self.n_iterations = 5000              # Number of MCMC steps per walker
        self.n_walkers = 50                   # Number of ensemble walkers
        self.n_temps = 1                      # Number of parallel tempering temperatures
        self.burnin = 0                       # Burnin steps (applied during analysis)
        
        # Proposal settings
        self.stretch_move_a = 2.0             # Stretch move parameter
        
        # Starting point configuration
        self.start_scatter_factor = 0.01      # Factor to scale Fisher errors for starting points
        self.fisher_prior_scale = 1000        # Prior width = n * Fisher uncertainty
        
        # Backend settings
        self.reset_backend = True             # WARNING: Will delete existing MCMC file!
        
        # Multiprocessing (CPU only)
        self.use_multiprocessing = True       # Use multiprocessing pool
        
        # Random seeds
        self.seed_number = 0                  # Master random seed
        
        # ==================== OUTPUT SETTINGS ====================
        
        # File naming
        self.output_tag = "MBH_HMs_case_1_tdi2_SNR_1137_AE"
        self.noise_tag = "w_noise"
        self.gap_tag = "w_window"  # "no_mask", "w_mask", or "w_window"
        
        # ==================== APPLY USER OVERRIDES ====================
        
        # Override any parameter passed as keyword argument
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown configuration parameter '{key}' ignored")
        
        # ==================== DERIVED QUANTITIES ====================
        
        # Recompute derived quantities after overrides
        self._compute_derived()
    
    def _compute_derived(self):
        """Compute derived quantities from primary parameters."""
        
        # Recompute array module if use_gpu changed
        if self.use_gpu:
            try:
                import cupy as cp
                self.xp = cp
            except ImportError:
                print("Warning: CuPy not available, falling back to NumPy")
                import numpy as np
                self.xp = np
                self.use_gpu = False
        else:
            import numpy as np
            self.xp = np
        
        # Recompute frequency spacing
        self.delta_f = 1 / self.t_obs
        
        # Recompute mass parameters
        self.M = self.m1 + self.m2
        self.q = self.m2 / self.m1
        
        # Time array
        self.time = np.arange(0, self.t_obs, self.delta_t)
        self.N = len(self.time)
        
        # Frequency array
        if self.f_max is None:
            self.f_max = 1 / (2 * self.delta_t)  # Nyquist frequency
        
        self.freq = self.xp.arange(self.f_min, self.f_max, self.delta_f)
        
        # Parameter vector for injection
        self.params = np.array([
            self.M, self.q, self.a1, self.a2, self.inc, self.dist_Gpc,
            self.phi_ref, self.lam, self.beta, self.psi, self.t_ref
        ])
        self.n_params = len(self.params)
        self.param_names = [
            'M', 'q', 'a1', 'a2', 'inc', 'dist_Gpc',
            'phi_ref', 'lam', 'beta', 'psi', 't_ref'
        ]
        
        # Waveform kwargs
        self.waveform_kwargs = {
            "freq": self.freq,
            "delta_t": self.delta_t,
            "delta_f": self.delta_f,
            "f_ref": self.f_ref,
            "modes": self.modes
        }
        
        # BBHx configuration
        self.amp_phase_kwargs = dict(run_phenomd=False)
        self.response_kwargs = dict(TDItag="AET", tdi2=True)
        
        # Output filename
        het_tag = "_heterodyne" if self.use_heterodyne else "_direct"
        self.output_filename = (
            f"{self.output_tag}_{self.noise_tag}_{self.gap_tag}_"
            f"seed_{self.seed_number}{het_tag}.h5"
        )
        self.output_filepath = os.path.join(self.mcmc_dir, self.output_filename)
        
        # Fisher matrix filename
        self.fisher_filename = "fisher_results.npy"
        self.fisher_filepath = os.path.join(self.fisher_dir, self.fisher_filename)
    
    def get_full_path(self, filename, directory='noise'):
        """
        Get full path for a file.
        
        Parameters
        ----------
        filename : str
            Filename to get path for
        directory : str, optional
            Which directory ('noise', 'mcmc', 'fisher', 'base')
            
        Returns
        -------
        str
            Full path to file
        """
        dir_map = {
            'noise': self.noise_dir,
            'mcmc': self.mcmc_dir,
            'fisher': self.fisher_dir,
            'base': self.base_dir
        }
        
        if directory not in dir_map:
            raise ValueError(f"Unknown directory '{directory}'. Choose from {list(dir_map.keys())}")
        
        return os.path.join(dir_map[directory], filename)
    
    def print_summary(self):
        """Print a summary of current configuration."""
        print("\n" + "=" * 80)
        print("MBH PARAMETER ESTIMATION CONFIGURATION")
        print("=" * 80)
        
        print("\n--- Computation ---")
        print(f"  Array backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")
        print(f"  Multiprocessing: {self.use_multiprocessing}")
        
        print("\n--- Analysis Mode ---")
        print(f"  Likelihood: {'Heterodyne (fast)' if self.use_heterodyne else 'Direct (slow)'}")
        print(f"  Gap handling: ", end="")
        if self.window:
            print(f"Windowing (Tukey tapers, {self.lobe_widths}s)")
        elif self.mask:
            print("Hard mask")
        elif self.no_mask:
            print("No gaps")
        else:
            print("Unknown")
        
        print("\n--- Data Parameters ---")
        print(f"  Channels: {self.channels}")
        print(f"  Observation time: {self.t_obs / (24*3600):.1f} days")
        print(f"  Time resolution: {self.delta_t} s")
        print(f"  Frequency range: {self.f_min:.2e} - {self.f_max:.2e} Hz")
        print(f"  Frequency resolution: {self.delta_f:.2e} Hz")
        print(f"  PSD file: {self.psd_filename}")
        
        print("\n--- Injection Parameters ---")
        print(f"  Total mass M: {self.M:.2e} M☉")
        print(f"  Mass ratio q: {self.q:.3f}")
        print(f"  Primary spin a1: {self.a1:.3f}")
        print(f"  Secondary spin a2: {self.a2:.3f}")
        print(f"  Inclination: {self.inc:.3f} rad ({np.degrees(self.inc):.1f}°)")
        print(f"  Distance: {self.dist_Gpc:.2f} Gpc")
        print(f"  Modes: {self.modes}")
        
        print("\n--- MCMC Settings ---")
        print(f"  Walkers: {self.n_walkers}")
        print(f"  Iterations: {self.n_iterations}")
        print(f"  Temperatures: {self.n_temps}")
        print(f"  Output file: {self.output_filename}")
        
        print("\n--- Directories ---")
        print(f"  Base: {self.base_dir}")
        print(f"  MCMC output: {self.mcmc_dir}")
        print(f"  Fisher matrices: {self.fisher_dir}")
        
        print("\n" + "=" * 80 + "\n")


# ==================== CONVENIENCE FUNCTIONS ====================

def get_default_config(**kwargs):
    """
    Get default configuration with optional overrides.
    
    Parameters
    ----------
    **kwargs : dict
        Configuration parameters to override
        
    Returns
    -------
    Config
        Configuration object
        
    Examples
    --------
    >>> cfg = get_default_config()
    >>> cfg = get_default_config(use_gpu=True, n_walkers=100)
    """
    return Config(**kwargs)


def get_fast_test_config():
    """
    Get a fast configuration for testing (few iterations, direct likelihood).
    
    Returns
    -------
    Config
        Test configuration
    """
    return Config(
        n_iterations=10,
        n_walkers=10,
        use_heterodyne=False,  # Direct is more stable for quick tests
        check_snr=False,
        plot_waveform=False
    )


def get_production_config():
    """
    Get a production configuration for full MCMC runs.
    
    Returns
    -------
    Config
        Production configuration
    """
    return Config(
        n_iterations=10000,
        n_walkers=50,
        n_temps=1,
        use_heterodyne=True,
        use_multiprocessing=True,
        reset_backend=False  # Don't delete existing runs!
    )


# ==================== MAIN ====================

if __name__ == "__main__":
    # Print default configuration when run as script
    cfg = get_default_config()
    cfg.print_summary()
