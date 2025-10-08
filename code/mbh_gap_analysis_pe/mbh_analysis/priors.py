"""
Prior Distributions for MBH MCMC Sampling

This module provides functions for setting up priors for MCMC parameter estimation.

Functions
---------
create_mbh_priors : Create prior distributions for MBH parameters
create_fisher_based_priors : Create priors scaled by Fisher matrix uncertainties
"""

import numpy as np
from eryn.prior import ProbDistContainer, uniform_dist


def create_mbh_priors(M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
                      Delta_params=None, n_sigma=1000, use_cupy=False):
    """
    Create uniform prior distributions for MBH parameters.

    Parameters
    ----------
    M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref : float
        Central values for each parameter
    Delta_params : array, optional
        Uncertainties for each parameter (e.g., from Fisher matrix).
        If None, uses wide priors. Default: None
    n_sigma : float, optional
        Prior width in units of Delta_params. Default: 1000 (very wide)
    use_cupy : bool, optional
        Use CuPy for GPU acceleration. Default: False

    Returns
    -------
    ProbDistContainer
        Prior distribution object for eryn sampler

    Notes
    -----
    Parameter order: [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
    
    If Delta_params is provided, priors are: param ± n_sigma * Delta_param
    Otherwise, uses physically reasonable ranges.
    """

    if Delta_params is None:
        # Wide priors without Fisher matrix
        priors_in = {
            0: uniform_dist(0.5 * M, 2.0 * M),              # Total mass M
            1: uniform_dist(0.01, 1.0),                      # Mass ratio q
            2: uniform_dist(-1.0, 1.0),                      # Primary spin a1
            3: uniform_dist(-1.0, 1.0),                      # Secondary spin a2
            4: uniform_dist(0.0, np.pi),                     # Inclination
            5: uniform_dist(0.1 * dist_Gpc, 10 * dist_Gpc), # Distance
            6: uniform_dist(0.0, 2 * np.pi),                 # Reference phase
            7: uniform_dist(0.0, 2 * np.pi),                 # Longitude lambda
            8: uniform_dist(-np.pi/2, np.pi/2),              # Latitude beta
            9: uniform_dist(0.0, np.pi),                     # Polarization psi
            10: uniform_dist(0.0, 4 * 365.25 * 24 * 3600)   # Reference time (4 years)
        }
    else:
        # Fisher-based priors
        priors_in = {
            0: uniform_dist(M - n_sigma * Delta_params[0], 
                          M + n_sigma * Delta_params[0]),
            1: uniform_dist(q - n_sigma * Delta_params[1], 
                          q + n_sigma * Delta_params[1]),
            2: uniform_dist(a1 - n_sigma * Delta_params[2], 
                          a1 + n_sigma * Delta_params[2]),
            3: uniform_dist(a2 - n_sigma * Delta_params[3], 
                          a2 + n_sigma * Delta_params[3]),
            4: uniform_dist(inc - n_sigma * Delta_params[4], 
                          inc + n_sigma * Delta_params[4]),
            5: uniform_dist(dist_Gpc - n_sigma * Delta_params[5], 
                          dist_Gpc + n_sigma * Delta_params[5]),
            6: uniform_dist(phi_ref - n_sigma * Delta_params[6], 
                          phi_ref + n_sigma * Delta_params[6]),
            7: uniform_dist(lam - n_sigma * Delta_params[7], 
                          lam + n_sigma * Delta_params[7]),
            8: uniform_dist(beta - n_sigma * Delta_params[8], 
                          beta + n_sigma * Delta_params[8]),
            9: uniform_dist(psi - n_sigma * Delta_params[9], 
                          psi + n_sigma * Delta_params[9]),
            10: uniform_dist(t_ref - n_sigma * Delta_params[10], 
                           t_ref + n_sigma * Delta_params[10])
        }

    return ProbDistContainer(priors_in, use_cupy=use_cupy)


def create_fisher_based_priors(true_params, cov_matrix, n_sigma=1000, use_cupy=False):
    """
    Create priors based on Fisher matrix covariance.

    Parameters
    ----------
    true_params : array
        True parameter values [M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref]
    cov_matrix : array
        Covariance matrix from Fisher matrix, shape (11, 11)
    n_sigma : float, optional
        Prior width in units of Fisher uncertainties. Default: 1000
    use_cupy : bool, optional
        Use CuPy for GPU. Default: False

    Returns
    -------
    ProbDistContainer
        Prior distributions for MCMC sampling

    Notes
    -----
    Extracts diagonal uncertainties from covariance matrix.
    """

    # Extract parameter uncertainties
    Delta_params = np.sqrt(np.diag(cov_matrix))

    # Create priors
    M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref = true_params

    return create_mbh_priors(
        M, q, a1, a2, inc, dist_Gpc, phi_ref, lam, beta, psi, t_ref,
        Delta_params=Delta_params,
        n_sigma=n_sigma,
        use_cupy=use_cupy
    )


def create_starting_points(true_params, cov_matrix, n_walkers, scatter_factor=0.01, seed=None):
    """
    Create starting points for MCMC walkers.

    Parameters
    ----------
    true_params : array
        True parameter values
    cov_matrix : array
        Covariance matrix from Fisher matrix
    n_walkers : int
        Number of MCMC walkers
    scatter_factor : float, optional
        Factor to scale Fisher uncertainties for initial scatter. Default: 0.01
    seed : int, optional
        Random seed. Default: None

    Returns
    -------
    array
        Starting points, shape (n_walkers, n_params)

    Notes
    -----
    Scatters walkers around true values with Gaussian distribution.
    """

    if seed is not None:
        np.random.seed(seed)

    n_params = len(true_params)
    Delta_params = np.sqrt(np.diag(cov_matrix))

    # Create starting points
    start = np.zeros((n_walkers, n_params))
    for i in range(n_params):
        start[:, i] = true_params[i] + scatter_factor * Delta_params[i] * np.random.randn(n_walkers)

    return start
