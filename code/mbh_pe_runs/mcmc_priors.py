
import numpy as np
import cupy as cp
import sys
sys.path.append("/home/ad/burkeol/work/EMRI_Lensing/utils")
from settings import (M, mu, a, p0, e0, iota0, Y0,
                      dist, Phi_phi0, Phi_theta0,
                      Phi_r0, qS, phiS, qK, phiK)

use_gpu = True
if use_gpu:
    xp = cp
else:
    xp = np

M_low =  1e4 
M_high = 1e7

q_low = 0
q_high = 20

a_low = 0 
a_high = 1

np.array([M, q, a1, a2, inc, dist_Gpc, phi_ref, lam,beta,psi,t_ref]) 

inc_low  = 0.0
inc_high = np.pi

dist_low = 0.1
dist_high = 200

phi_ref_low = -2*np.pi 
phi_ref_high = 2*np.pi

lam_low = 0 
lam_high = 2*xp.pi

beta_low = -xp.pi/2
beta_high = xp.pi/2

psi_low = 0.0
psi_high = xp.pi

def lprior_M(M,M_low, M_high):
    if M < M_low or M > M_high:
        print("M has failed")
        return -xp.inf
    else:
        return 0

def lprior_q(q,q_low, q_high):
    if q < q_low or q > q_high:
        print("q has failed")
        return -xp.inf
    else:
        return 0

def lprior_a1(a1,a_low, a_high):
    if a1 < a_low or a1 > a_high:
        print("a1 has failed")
        return -xp.inf
    else:
        return 0

def lprior_a2(a2,a_low, a_high):
    if a2 < a_low or a2 > a_high:
        print("a2 has failed")
        return -xp.inf
    else:
        return 0

def lprior_inc(inc,inc_low, inc_high):
    if inc < inc_low or inc > inc_high:
        print("inc has failed")
        return -xp.inf
    else:
        return 0
    
def lprior_dist(dist,dist_low, dist_high):
    if dist < dist_low or dist > dist_high:
        print("dist has failed")
        return -xp.inf
    else:
        return 0

def lprior_phi_ref(phi_ref,phi_ref_low, phi_ref_high):
    if phi_ref < phi_ref_low or phi_ref > phi_ref_high:
        print("phi_ref has failed")
        return -xp.inf
    else:
        return 0


def lprior_lam(lam, lam_low, lam_high):
    if lam < lam_low or lam > lam_high:
        print("the parameter", lam, "has failed")
        return -xp.inf
    else:
        return 0

def lrior_lam(lam, lam_low, lam_high):
    if lam < lam_low or lam > lam_high:
        print("the parameter", lam, "has failed")
        return -xp.inf
    else:
        return 0
def lprior_mu_pos(mu_pos,mu_pos_low, mu_pos_high):
    if mu_pos < mu_pos_low  or mu_pos > mu_pos_high:
        print("mu_pos has failed")
        return -xp.inf
    else:
        return 0

def lprior_mu_neg(mu_neg, mu_neg_low, mu_neg_high):
    if mu_neg < mu_neg_low  or mu_neg > mu_neg_high:
        print("mu_neg has failed")
        return -xp.inf
    else:
        return 0

def lprior_y(y, y_low, y_high):
    if y < y_low  or y > y_high:
        print("y has failed")
        return -xp.inf
    else:
        return 0


def lprior_td(td, td_low, td_high):
    if td < td_low  or td > td_high:
        print("td has failed")
        return -xp.inf
    else:
        return 0

def lprior(params):
    log_prior = (lprior_M(params[0],M_low, M_high)+ 
                lprior_mu(params[1],mu_low, mu_high) +
                lprior_a(params[2],a_low, a_high) + 
                lprior_p0(params[3],p0_low, p0_high) +
                lprior_e0(params[4],e0_low,e0_high)  + 
                lprior_Y0(params[5],Y0_low, Y0_high) +
                # lprior_D(params[6],D_low, D_high) +
                lprior_angle(params[6], 0, cp.pi) +
                lprior_angle(params[7],angle_low, angle_high) +
                lprior_angle(params[8],0, cp.pi) +
                lprior_angle(params[9],angle_low, angle_high) +
                lprior_angle(params[10],angle_low, angle_high) +
                lprior_angle(params[11],angle_low, angle_high) + 
                lprior_angle(params[12],angle_low, angle_high) + 
                lprior_y(params[13], y_low, y_high) +
                lprior_td(params[14],td_low, td_high))
    if xp.isinf(log_prior):
        return -xp.inf
    return log_prior
