"""
MBH Analysis Package

A modular package for Massive Black Hole (MBH) parameter estimation with LISA.

Modules
-------
waveforms
    Waveform generation for MBH binaries
likelihood
    Likelihood computation (heterodyned and direct)
noise
    PSD handling and noise generation
windows
    Gap windowing functions
fisher
    Fisher matrix computation
priors
    Prior distributions for MCMC

Usage
-----
>>> from mbh_analysis import waveforms, likelihood, noise
>>> # Your analysis code here
"""

__version__ = "2.0.0"
__author__ = "Ollie Burke"

# Import key functions for convenience
from .waveforms import MBH_f, GapBBHWaveformFD
from .likelihood import GapLikelihood
from .noise import inner_prod, generate_colored_noise, load_psd_from_file
from .windows import create_gap_window
from .fisher import build_fish_matrix, load_fisher_matrix

__all__ = [
    'MBH_f',
    'GapBBHWaveformFD',
    'GapLikelihood',
    'inner_prod',
    'generate_colored_noise',
    'load_psd_from_file',
    'create_gap_window',
    'build_fish_matrix',
    'load_fisher_matrix',
]
