# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python scripts for gravitational wave data analysis, specifically for searching and parameter estimation in the presence of glitches. The code is designed to work with LISA (Laser Interferometer Space Antenna) data and implements the methods described in https://arxiv.org/pdf/2505.19870.

## Key Dependencies

The project relies on several specialized gravitational wave analysis packages:
- `bbhx`: Binary black hole waveform generation and LISA response functions
- `eryn`: MCMC ensemble sampling with advanced moves and reversible jump capabilities
- `lisatools`: LISA-specific analysis utilities including glitch modeling and sensitivity curves
- `chainconsumer`: Plotting and analysis of MCMC chains
- Standard scientific Python stack: `numpy`, `scipy`, `matplotlib`, `cupy` (optional for GPU acceleration)

## Core Components

### Main Analysis Scripts
1. **Spritz Data Analysis**:
   - `LDC2_spritz_analysis_shorter_segments.py`: Search analysis for shorter data segments
   - `LDC2_spritz_analysis_new.py`: Parameter estimation for Spritz data

2. **Glitch Modeling**:
   - `glitch_shapelet_analytical_waveform.py`: Analytical glitch waveform generation (time/frequency domain)
   - `glitch_fitting_freq_domain_RJ.py`: Glitch distribution fitting with reversible jump MCMC
   - `glitch_fitting_mbhb_with_noise.py`: Parameter estimation with glitches, massive black hole binaries, and noise

3. **Supporting Utilities**:
   - `synthetic_noise_generator.py`: Synthetic noise and PSD generation
   - `max_matching_glitch_MBHB.py`: Maximum overlap calculations between MBHB and glitch signals
   - `group_stretch_proposal.py`: Custom MCMC proposal move for glitch searches

### Architecture Notes

- All scripts are standalone Python files without a formal package structure
- GPU acceleration is optionally supported via CuPy (falls back to NumPy if unavailable)
- The codebase uses the Eryn MCMC framework with custom moves for glitch parameter estimation
- Glitch modeling uses shapelet basis functions in both time and frequency domains
- LISA TDI (Time Delay Interferometry) response is handled through bbhx package

## Development

Since this is a research codebase with standalone scripts rather than a formal package:
- Each Python file can be run independently
- No build system or test framework is present
- Dependencies must be manually installed (Eryn, lisaanalysis, BBhx from https://github.com/mikekatz04)
- GPU usage is encouraged but optional

## Important Files

- External data dependency: `glitch_params_all_PRDLPF.h5` (not publicly available, contact authors)
- All `.py` files are analysis scripts - refer to README.md for specific use cases