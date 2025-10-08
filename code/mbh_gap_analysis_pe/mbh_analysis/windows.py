"""
Gap Windowing Functions for LISA Data Analysis

This module provides functions for creating gap windows with smooth tapers.

Functions
---------
create_gap_window : Create gap window with Tukey tapers

Constants
---------
gap_definitions : Dictionary of gap types and properties
taper_defs : Dictionary of taper configurations  
"""

import numpy as np


# ========== CONFIG FOR GAP AND TAPER DEFINITIONS ========== #

gap_definitions = {
    "planned": {
        "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
    },
    "unplanned": {
        "Aliens": {"rate_per_year": 100, "duration_hr": 10/60},
    }
}

taper_defs = {
    "planned": {
        "antenna repointing": {"lobe_lengths_hr": 10/60},
    },
    "unplanned": {
        "Aliens": {"lobe_lengths_hr": 10/60},
    }
}

include_planned = True
include_unplanned = True
planned_seed = 1234
unplanned_seed = 4321


# ========== GAP WINDOW FUNCTION WITH TUKEY TAPERS ========== #

def create_gap_window(time_array, gap_centers, gap_widths, lobe_widths=0.0, use_gpu=False):
    """
    Create a gap window function with symmetric Tukey-style tapers.
    
    This function creates a window where w(t) = 0 inside gaps (data loss) and w(t) = 1
    elsewhere (good data), with smooth cosine tapers at the gap edges.
    
    Parameters
    ----------
    time_array : array-like
        Time array in seconds. Shape: (N_time,)
    gap_centers : list or array-like
        Center times of each gap in seconds. Can be a single value or list.
        Example: [1000.0, 5000.0, 10000.0]
    gap_widths : list or array-like
        Width of each gap in seconds - the exact duration where w = 0.
        Can be a single value (applied to all gaps) or list matching gap_centers.
        Example: [3*3600, 2*3600, 4*3600] for 3hr, 2hr, 4hr gaps
    lobe_widths : float or list, optional
        Width of each taper lobe in seconds (symmetric on both sides of gap).
        Can be a single value (applied to all gaps) or list matching gap_centers.
        Example: 0.3*3600 for 18-minute tapers. Default: 0.0 (no tapers)
    use_gpu : bool, optional
        If True, use cupy for GPU acceleration. Default: False
    
    Returns
    -------
    window : array
        Gap window function with same shape as time_array.
        Values: w = 1 (good data), w = 0 (gap), smooth transitions at edges
    
    Notes
    -----
    Gap structure for each gap:
    
    ```
    ├──────────┬───────────┬──────────┬───────────┬──────────┤
    │  w = 1   │ Left lobe │  w = 0   │Right lobe │  w = 1   │
    │ (data)   │ 1 → 0     │  (gap)   │  0 → 1    │ (data)   │
    └──────────┴───────────┴──────────┴───────────┴──────────┘
               ↑           ↑          ↑           ↑
        gap_start    actual_gap  actual_gap  gap_end
          - lobe       start        end      + lobe
    ```
    
    - Gap core: [center - width/2, center + width/2] has w = 0
    - Left lobe: [gap_start - lobe_width, gap_start] transitions 1 → 0
    - Right lobe: [gap_end, gap_end + lobe_width] transitions 0 → 1
    - Total affected time: gap_width + 2 * lobe_width
    
    The taper uses a symmetric Tukey window with cosine rolloff:
        w(t) = 0.5 * (1 + cos(π * (t - t_start) / lobe_width))
    
    Examples
    --------
    >>> import numpy as np
    >>> 
    >>> # Create time array for 2 months
    >>> T_obs = 2 * 30 * 24 * 3600  # seconds
    >>> dt = 5.0
    >>> time = np.arange(0, T_obs, dt)
    >>> 
    >>> # Create 3 gaps: 3hr, 2hr, 4hr with 18-minute tapers
    >>> gap_centers = [10*24*3600, 30*24*3600, 50*24*3600]
    >>> gap_widths = [3*3600, 2*3600, 4*3600]
    >>> lobe_widths = 0.3*3600  # 18 minutes
    >>> 
    >>> window = create_gap_window(time, gap_centers, gap_widths, lobe_widths)
    >>> 
    >>> # Single gap with no tapers (hard mask)
    >>> window_hard = create_gap_window(time, 1.5e6, 7200, lobe_widths=0.0)
    """
    
    # Select array backend
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
        except ImportError:
            print("Warning: CuPy not available, falling back to NumPy")
            xp = np
    else:
        xp = np

    # Convert time_array to appropriate backend
    time_array = xp.asarray(time_array)
    
    # Normalize inputs to lists
    if not isinstance(gap_centers, (list, tuple, np.ndarray)):
        gap_centers = [gap_centers]
    if not isinstance(gap_widths, (list, tuple, np.ndarray)):
        gap_widths = [gap_widths] * len(gap_centers)
    if not isinstance(lobe_widths, (list, tuple, np.ndarray)):
        lobe_widths = [lobe_widths] * len(gap_centers)
    
    # Check consistency
    if len(gap_widths) != len(gap_centers):
        raise ValueError(f"gap_widths length ({len(gap_widths)}) must match gap_centers ({len(gap_centers)})")
    if len(lobe_widths) != len(gap_centers):
        raise ValueError(f"lobe_widths length ({len(lobe_widths)}) must match gap_centers ({len(gap_centers)})")
    
    # Start with window = 1 everywhere (all good data)
    window = xp.ones_like(time_array)
    
    # Apply each gap
    for gap_center, gap_width, lobe_width in zip(gap_centers, gap_widths, lobe_widths):
        # Gap core boundaries [center - width/2, center + width/2]
        gap_start = gap_center - gap_width / 2.0
        gap_end = gap_center + gap_width / 2.0
        
        # Total affected region includes tapers
        total_start = gap_start - lobe_width
        total_end = gap_end + lobe_width
        
        # Find indices in affected region
        affected = (time_array >= total_start) & (time_array <= total_end)
        
        if not xp.any(affected):
            # No overlap with time array
            continue
        
        # Extract affected times
        t_affected = time_array[affected]
        
        # Compute window values for affected region
        w_affected = xp.ones_like(t_affected)
        
        # Left taper: [gap_start - lobe_width, gap_start]
        if lobe_width > 0:
            left_taper = (t_affected >= total_start) & (t_affected < gap_start)
            if xp.any(left_taper):
                # Cosine taper from 1 → 0
                phase = (t_affected[left_taper] - total_start) / lobe_width
                w_affected[left_taper] = 0.5 * (1 + xp.cos(xp.pi * (1 - phase)))
        
        # Gap core: [gap_start, gap_end] → w = 0
        core = (t_affected >= gap_start) & (t_affected <= gap_end)
        w_affected[core] = 0.0
        
        # Right taper: [gap_end, gap_end + lobe_width]
        if lobe_width > 0:
            right_taper = (t_affected > gap_end) & (t_affected <= total_end)
            if xp.any(right_taper):
                # Cosine taper from 0 → 1
                phase = (t_affected[right_taper] - gap_end) / lobe_width
                w_affected[right_taper] = 0.5 * (1 + xp.cos(xp.pi * (1 + phase)))
        
        # Apply this gap's window (multiply to handle overlapping gaps)
        window[affected] *= w_affected
    
    return window
