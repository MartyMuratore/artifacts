# Legacy Files (Archived)

This directory contains the original utility files that have been superseded by the new `mbh_analysis/` package structure.

## Files in This Directory

### Original Utility Files (Now in `mbh_analysis/`)

| Old File | New Location | Description |
|----------|--------------|-------------|
| `mbh_utils.py` | `mbh_analysis/waveforms.py` | Waveform generation (`MBH_f`, `GapBBHWaveformFD`) |
| `noise_utils.py` | `mbh_analysis/noise.py` | PSD handling and noise generation |
| `window_func.py` | `mbh_analysis/windows.py` | Gap windowing with Tukey tapers |
| `gap_heterodyne_likelihood.py` | `mbh_analysis/likelihood.py` | Likelihood computation |

### Legacy Scripts

| File | Status | Notes |
|------|--------|-------|
| `mcmc_MBH.py` | Deprecated | Use `mcmc_MBH_heterodyne.py` instead |

## Why These Files Are Here

These files have been reorganized into the `mbh_analysis/` package for better code organization:

✅ **Better organization** - Related functions grouped into logical modules  
✅ **Cleaner imports** - `from mbh_analysis import ...`  
✅ **Better documentation** - Full docstrings and module README  
✅ **Professional structure** - Standard Python package layout  

## Can I Delete These Files?

**Not yet recommended!** Keep them for:
- Reference during migration period
- Comparison with new implementation
- Rollback option if needed
- Historical record of development

After you've validated the new package works for all your use cases, you can:
1. Archive them in git history
2. Delete from working directory
3. Or keep indefinitely (they're small)

## Using the New Package

Instead of:
```python
from mbh_utils import MBH_f
from noise_utils import inner_prod
from window_func import create_gap_window
from gap_heterodyne_likelihood import GapLikelihood
```

Use:
```python
from mbh_analysis import MBH_f, inner_prod, create_gap_window, GapLikelihood
```

## More Information

- **Package documentation:** `../mbh_analysis/README.md`
- **Migration guide:** `../README.md`
- **Configuration guide:** `../README_CONFIG.md`

---

**Status:** Archived but preserved  
**Date:** October 8, 2025  
**Reason:** Replaced by `mbh_analysis/` package
