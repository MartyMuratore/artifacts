# MBH Analysis Package

A modular Python package for Massive Black Hole (MBH) binary parameter estimation with LISA.

## Package Structure

```
mbh_analysis/
├── __init__.py          Package initialization, convenient imports
├── waveforms.py         Waveform generation (MBH_f, GapBBHWaveformFD)
├── likelihood.py        Likelihood computation (GapLikelihood)
├── noise.py             PSD handling and noise generation
├── windows.py           Gap windowing functions
├── fisher.py            Fisher matrix computation
└── priors.py            Prior distributions for MCMC
```

## Quick Start

### Basic Usage

```python
from mbh_analysis import MBH_f, build_fish_matrix, create_gap_window
from config import get_default_config

# Load configuration
cfg = get_default_config()

# Generate waveform
waveform = MBH_f(wave_gen, cfg.M, cfg.q, cfg.a1, cfg.a2, 
                 cfg.inc, cfg.dist_Gpc, cfg.phi_ref,
                 cfg.lam, cfg.beta, cfg.psi, cfg.t_ref,
                 freq=cfg.freq, f_ref=cfg.f_ref, modes=cfg.modes)
```

### Using the Package

```python
# Method 1: Import specific functions
from mbh_analysis import MBH_f, create_gap_window, inner_prod

# Method 2: Import modules
from mbh_analysis import waveforms, noise, fisher

# Method 3: Import everything
import mbh_analysis as mba
```

## Module Documentation

### `waveforms.py` - Waveform Generation

**Functions:**
- `MBH_f(wave_gen, M, q, ...)` - Generate MBH waveforms
- `get_array_module(array)` - Detect numpy vs cupy

**Classes:**
- `GapBBHWaveformFD` - Gap-aware waveform generator

**Example:**
```python
from mbh_analysis import MBH_f
from bbhx.waveformbuild import BBHWaveformFD

wave_gen = BBHWaveformFD(...)
waveform = MBH_f(wave_gen, M=2e6, q=0.5, a1=0.7, a2=0.8,
                 inc=np.pi/3, dist_Gpc=40.0, phi_ref=1.2,
                 lam=1.0, beta=-0.3, psi=np.pi/6, t_ref=1e6,
                 freq=freq_array, f_ref=0.0, modes=[(2,2)])
```

### `likelihood.py` - Likelihood Computation

**Classes:**
- `GapLikelihood` - Heterodyned likelihood for fast MCMC

**Example:**
```python
from mbh_analysis import GapLikelihood

likelihood_obj = GapLikelihood(
    wave_gen=wave_gen,
    data_channels=data_f_AE,
    data_freqs=freq,
    gap_window_array=gap_window,
    delta_t=5.0,
    delta_f=1/T_obs,
    N=N_time,
    PSD_interp=PSD_AE_interp,
    reference_params=params,
    f_ref=0.0,
    modes=modes,
    use_heterodyned=True
)

# Use in MCMC
log_likelihood = likelihood_obj(test_params)
```

### `noise.py` - Noise Utilities

**Functions:**
- `inner_prod(signal_1, signal_2, PSD, delta_f)` - Inner product
- `load_psd_from_file(psd_file)` - Load PSD interpolator
- `generate_colored_noise(variance, delta_t, seed)` - Generate noise
- `pad_to_length(array, N)` - Pad arrays

**Example:**
```python
from mbh_analysis import inner_prod, load_psd_from_file, generate_colored_noise

# Load PSD
PSD_interp = load_psd_from_file('tdi2_AE_w_background.npy')
PSD = PSD_interp(freq)

# Compute SNR
snr_squared = inner_prod(waveform, waveform, PSD, delta_f)
snr = np.sqrt(snr_squared)

# Generate noise
noise = generate_colored_noise(PSD / (4*delta_f), delta_t, seed=42)
```

### `windows.py` - Gap Windowing

**Functions:**
- `create_gap_window(time, gap_centers, gap_widths, lobe_widths)` - Create gap window

**Constants:**
- `gap_definitions` - Gap type configurations
- `taper_defs` - Taper configurations

**Example:**
```python
from mbh_analysis import create_gap_window

# Create 3 gaps with Tukey tapers
time = np.arange(0, T_obs, delta_t)
gap_centers = [2.6e6, 2.62e6, 2.64e6]
gap_widths = [3*3600, 2*3600, 4*3600]  # 3hr, 2hr, 4hr
lobe_widths = 5*60  # 5-minute tapers

window = create_gap_window(time, gap_centers, gap_widths, lobe_widths)
```

### `fisher.py` - Fisher Matrix

**Functions:**
- `build_fish_matrix(wave_gen, M, q, ...)` - Compute Fisher matrix
- `load_fisher_matrix(file_path)` - Load from file

**Example:**
```python
from mbh_analysis import build_fish_matrix, load_fisher_matrix

# Compute Fisher matrix
Fish = build_fish_matrix(wave_gen, M, q, a1, a2, inc, dist_Gpc,
                        phi_ref, lam, beta, psi, t_ref,
                        window_func=gap_window,
                        delta_f=delta_f, delta_t=delta_t,
                        PSD=PSD, freq=freq, f_ref=0.0, modes=modes)

# Get parameter uncertainties
Cov = np.linalg.inv(Fish)
uncertainties = np.sqrt(np.diag(Cov))

# Load previous results
results = load_fisher_matrix('fisher_results.npy')
```

### `priors.py` - Prior Distributions

**Functions:**
- `create_mbh_priors(M, q, ..., Delta_params)` - Create priors
- `create_fisher_based_priors(true_params, cov_matrix)` - Fisher-based priors
- `create_starting_points(true_params, cov_matrix, n_walkers)` - MCMC starting points

**Example:**
```python
from mbh_analysis import create_fisher_based_priors, create_starting_points

# Create priors from Fisher matrix
priors = create_fisher_based_priors(
    true_params=params,
    cov_matrix=Cov_matrix,
    n_sigma=1000,
    use_cupy=False
)

# Create starting points for walkers
start = create_starting_points(
    true_params=params,
    cov_matrix=Cov_matrix,
    n_walkers=50,
    scatter_factor=0.01
)
```

## Migration from Old Code

### Old Way (Phase 1)
```python
from mbh_utils import MBH_f, build_fish_matrix
from noise_utils import inner_prod, load_psd_from_file
from window_func import create_gap_window
from gap_heterodyne_likelihood import GapLikelihood
```

### New Way (Phase 2)
```python
from mbh_analysis import (
    MBH_f, build_fish_matrix,
    inner_prod, load_psd_from_file,
    create_gap_window, GapLikelihood
)
```

**Benefits:**
- Single import statement
- Clear module organization
- Better discoverability
- Easier to maintain

## Backward Compatibility

**Old files still work!** The original files are preserved:
- `mbh_utils.py` ✓ Still available
- `noise_utils.py` ✓ Still available
- `window_func.py` ✓ Still available
- `gap_heterodyne_likelihood.py` ✓ Still available

**Migration is optional** - take your time!

## Complete Example

```python
import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from mbh_analysis import (
    MBH_f, GapLikelihood, create_gap_window,
    load_psd_from_file, inner_prod, build_fish_matrix,
    create_fisher_based_priors, create_starting_points
)
from config import get_default_config

# 1. Configuration
cfg = get_default_config()

# 2. Set up waveform generator
wave_gen = BBHWaveformFD(
    amp_phase_kwargs=cfg.amp_phase_kwargs,
    response_kwargs=cfg.response_kwargs
)

# 3. Generate gap window
time = cfg.time
window = create_gap_window(time, cfg.gap_centers, 
                           cfg.gap_widths, cfg.lobe_widths)

# 4. Load PSD
PSD_interp = load_psd_from_file(cfg.get_full_path(cfg.psd_filename, 'noise'))
PSD = PSD_interp(cfg.freq)

# 5. Generate waveform
waveform = MBH_f(wave_gen, cfg.M, cfg.q, cfg.a1, cfg.a2,
                 cfg.inc, cfg.dist_Gpc, cfg.phi_ref,
                 cfg.lam, cfg.beta, cfg.psi, cfg.t_ref,
                 freq=cfg.freq, f_ref=cfg.f_ref, modes=cfg.modes)

# 6. Compute SNR
snr2 = np.sum([inner_prod(waveform[i], waveform[i], PSD[i], cfg.delta_f) 
               for i in range(cfg.n_channels)])
print(f"SNR: {np.sqrt(snr2):.1f}")

# 7. Build Fisher matrix
Fish = build_fish_matrix(wave_gen, cfg.M, cfg.q, cfg.a1, cfg.a2,
                        cfg.inc, cfg.dist_Gpc, cfg.phi_ref,
                        cfg.lam, cfg.beta, cfg.psi, cfg.t_ref,
                        window_func=window,
                        delta_f=cfg.delta_f, delta_t=cfg.delta_t,
                        PSD=PSD, freq=cfg.freq, f_ref=cfg.f_ref,
                        modes=cfg.modes)
Cov = np.linalg.inv(Fish)

# 8. Set up MCMC
priors = create_fisher_based_priors(cfg.params, Cov)
start = create_starting_points(cfg.params, Cov, cfg.n_walkers)

# 9. Create likelihood
likelihood = GapLikelihood(
    wave_gen=wave_gen,
    data_channels=data,
    data_freqs=cfg.freq,
    gap_window_array=window,
    delta_t=cfg.delta_t,
    delta_f=cfg.delta_f,
    N=cfg.N,
    PSD_interp=PSD_interp,
    reference_params=cfg.params,
    use_heterodyned=True
)

# 10. Run MCMC (with eryn)
# ... sampler setup and run_mcmc ...
```

## Testing

Test the package installation:

```bash
python -c "from mbh_analysis import MBH_f; print('✓ Import successful')"
```

Test all modules:

```bash
python -c "from mbh_analysis import *; print('✓ All imports successful')"
```

## Advantages

1. **Organization** - Related functions grouped together
2. **Discovery** - Easy to find what you need
3. **Maintenance** - Changes in one place
4. **Testing** - Each module can be tested independently
5. **Documentation** - Clear module boundaries
6. **Imports** - Cleaner import statements
7. **Backward Compatible** - Old code still works

## See Also

- `config.py` - Configuration system (Phase 1)
- `STRUCTURE.md` - Visual guide to organization
- `README_CONFIG.md` - Configuration documentation
- `PHASE2_COMPLETE.md` - Phase 2 implementation notes

---

**Package ready to use!** 🎉
