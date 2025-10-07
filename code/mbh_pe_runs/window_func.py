
#### ========== CONFIG FILE FOR GAP AND TAPER DEFINITIONS ========== ####

gap_definitions = {
    "planned": {
        "antenna repointing": {"rate_per_year": 26, "duration_hr": 3.3},
    },
    "unplanned": {
        "Aliens": {"rate_per_year": 100, "duration_hr": 10/60},
    }
}

# Set up taper information

taper_defs = {
    "planned": {
        "antenna repointing": {"lobe_lengths_hr": 10/60},
    },
    "unplanned": {
        "Aliens": {"lobe_lengths_hr": 10/60},
    }
}

include_planned=True
include_unplanned=True
planned_seed = 1234
unplanned_seed = 4321


#### ========== GAP WINDOW FUNCTION WITH TUKEY TAPERS ========== ####

def create_gap_window(time_array, gap_centers, gap_widths, lobe_widths = 0.0, use_gpu=False):
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
    lobe_widths : float or list
        Width of each taper lobe in seconds (symmetric on both sides of gap).
        Can be a single value (applied to all gaps) or list matching gap_centers.
        Example: 0.3*3600 for 18-minute tapers, or [0.5*3600, 0.2*3600, 0.4*3600]
    use_gpu : bool, optional
        If True, use cupy for GPU acceleration. If False, use numpy. Default: False
    
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
    >>> gap_centers = [10*24*3600, 30*24*3600, 50*24*3600]  # at 10, 30, 50 days
    >>> gap_widths = [3*3600, 2*3600, 4*3600]  # gap durations
    >>> lobe_widths = 0.3*3600  # 18-minute tapers (same for all)
    >>> 
    >>> window = create_gap_window(time, gap_centers, gap_widths, lobe_widths)
    >>> 
    >>> # Apply to waveform
    >>> waveform_gapped = waveform * window
    >>> 
    >>> # With different taper widths for each gap
    >>> lobe_widths = [0.5*3600, 0.2*3600, 0.4*3600]  # different for each
    >>> window = create_gap_window(time, gap_centers, gap_widths, lobe_widths)
    """
    
    # Select array module based on GPU flag
    if use_gpu:
        try:
            import cupy as cp
            xp = cp
            time_array = cp.asarray(time_array)
        except ImportError:
            print("Warning: CuPy not available, falling back to NumPy")
            import numpy as np
            xp = np
    else:
        import numpy as np
        xp = np
    
    # Ensure inputs are arrays
    time_array = xp.asarray(time_array)
    gap_centers = xp.atleast_1d(xp.asarray(gap_centers))
    gap_widths = xp.atleast_1d(xp.asarray(gap_widths))
    lobe_widths = xp.atleast_1d(xp.asarray(lobe_widths))
    
    n_gaps = len(gap_centers)
    
    # Handle scalar inputs - broadcast to all gaps
    if len(gap_widths) == 1 and n_gaps > 1:
        gap_widths = xp.ones(n_gaps) * gap_widths[0]
    if len(lobe_widths) == 1 and n_gaps > 1:
        lobe_widths = xp.ones(n_gaps) * lobe_widths[0]
    
    # Validate inputs
    if len(gap_widths) != n_gaps:
        raise ValueError(f"gap_widths length ({len(gap_widths)}) must match "
                        f"gap_centers length ({n_gaps}) or be scalar")
    if len(lobe_widths) != n_gaps:
        raise ValueError(f"lobe_widths length ({len(lobe_widths)}) must match "
                        f"gap_centers length ({n_gaps}) or be scalar")
    
    # Initialize window to all ones (good data everywhere)
    window = xp.ones_like(time_array, dtype=float)
    
    # Process each gap
    for i in range(n_gaps):
        center = gap_centers[i]
        width = gap_widths[i]
        lobe = lobe_widths[i]
        
        # Define gap boundaries
        gap_start = center - width / 2.0  # Start of w=0 region
        gap_end = center + width / 2.0    # End of w=0 region
        
        # Define taper boundaries (extended by lobe width on each side)
        taper_start = gap_start - lobe    # Start of left taper
        taper_end = gap_end + lobe        # End of right taper
        
        # Create masks for different regions
        in_gap_core = (time_array >= gap_start) & (time_array <= gap_end)
        in_left_taper = (time_array >= taper_start) & (time_array < gap_start)
        in_right_taper = (time_array > gap_end) & (time_array <= taper_end)
        
        # Set gap core to zero
        window[in_gap_core] = 0.0
        
        # Apply left taper: smooth transition from 1 → 0
        if xp.any(in_left_taper):
            t_left = time_array[in_left_taper]
            # Normalized position in taper: 0 at taper_start, 1 at gap_start
            x_left = (t_left - taper_start) / lobe
            # Cosine taper: 1 at start, 0 at end
            window[in_left_taper] = 0.5 * (1.0 + xp.cos(xp.pi * x_left))
        
        # Apply right taper: smooth transition from 0 → 1
        if xp.any(in_right_taper):
            t_right = time_array[in_right_taper]
            # Normalized position in taper: 0 at gap_end, 1 at taper_end
            x_right = (t_right - gap_end) / lobe
            # Cosine taper: 0 at start, 1 at end
            window[in_right_taper] = 0.5 * (1.0 - xp.cos(xp.pi * x_right))
    
    return window

