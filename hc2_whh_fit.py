#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-file Hc2(T) fitter (one-band and two-band WHH, dirty limit)

FIRST ARG: files (Python-list string OR comma/space-separated string)
SECOND ARG: mode ("orbital" | "pauli" | "2band_orbital" | "2band_pauli")

Examples:
  python hc2_whh_fit.py "[sc11.txt, sc33.txt]" pauli --Tc_fixed=2.061
  python hc2_whh_fit.py "sc11.txt,sc33.txt" 2band_orbital \
      --lam11=0.8 --lam22=0.3 --lam12=0.1 --lam21=0.1 --eta=0.25
  python hc2_whh_fit.py "sc11.txt sc33.txt" 2band_pauli \
      --alpha1_bounds=0,300 --alpha2_bounds=0,300 --lambda_so1_bounds=0,300 --lambda_so2_bounds=0,300

Default behavior: fit parameters (Tc, k, α’s, λ_so’s, and η) unless you FIX them via CLI.
Two-band couplings λ_ij are given by CLI values and are NOT fitted by default (change values as needed).

Outputs per input file:
  <stem>_<mode>_results.csv   (data + fitted monotone Hc2(T))
  <stem>_<mode>_params.csv    (fitted parameters, metrics, Hc2(0))
Plus combined:
  combined_<mode>_params.csv
  combined_<mode>_plot.png / .pdf   (data + fits, curves drawn from 0 K → Tc)
"""

import sys, ast, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import digamma as psi
from scipy.optimize import least_squares

# ============================== utilities ==============================

def parse_files_arg(s: str):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            lst = ast.literal_eval(s)
            if isinstance(lst, list):
                return [str(x).strip() for x in lst]
        except Exception:
            pass
    if ("," in s) or (" " in s):
        parts = [p for chunk in s.split(",") for p in chunk.split()]
        return [p.strip() for p in parts if p.strip()]
    return [s]

def enforce_strict_monotone(T_arr, Hpred):
    order = np.argsort(T_arr)
    Hp = Hpred[order].copy()
    for i in range(1, len(Hp)):
        if not (Hp[i] < Hp[i-1]):
            Hp[i] = np.nextafter(Hp[i-1], -np.inf)
    out = np.empty_like(Hpred); out[order] = Hp
    return out

def largest_root(T, F_fun, args, H_hint=None, H_data_max=None,
                 scans=300, iters=100, Hcap=5_000.0):
    if H_hint is None:
        Hhi = (1.5 * (H_data_max if H_data_max is not None else 50.0)) + 20.0
    else:
        Hhi = max(H_hint * 1.25, (H_data_max if H_data_max is not None else 0.0) + 10.0)
    Hhi = max(Hhi, 5.0)
    Hlo = 0.0
    def f(H): return F_fun(T, H, *args)
    flo = f(Hlo)
    if not (isinstance(flo, (int,float)) and np.isfinite(flo)):
        return float("nan")
    roots = []
    while Hhi <= Hcap:
        Hs = np.linspace(Hlo, Hhi, scans+1)
        Fs = np.array([f(h) for h in Hs])
        for i in range(scans):
            f1, f2 = Fs[i], Fs[i+1]
            if not (np.isfinite(f1) and np.isfinite(f2)): continue
            if f1 == 0.0: roots.append(Hs[i]); continue
            if f1 * f2 < 0.0:
                a,b = Hs[i], Hs[i+1]; fa,fb = f1,f2
                for _ in range(iters):
                    m = 0.5*(a+b); fm = f(m)
                    if not np.isfinite(fm): break
                    if abs(fm) < 1e-10: roots.append(m); break
                    if fa*fm < 0.0: b,fb = m,fm
                    else: a,fa = m,fm
                else:
                    roots.append(0.5*(a+b))
        if roots: return float(np.max(roots))
        Hhi *= 1.8
    return float("nan")

# ============================= WHH kernels =============================

# ---- One-band

def F1b_orbital_lhs(t: float, h: float) -> float:
    if t <= 0: return float("nan")
    return np.log(1.0/max(t,1e-12)) - (psi(0.5 + h/(2.0*t)) - psi(0.5))

def F1b_pauli_so_lhs(T: float, H: float, Tc: float, k: float, alpha: float, lam_so: float) -> float:
    if T <= 0 or Tc <= 0 or k <= 0 or alpha < 0 or lam_so < 0: return float("nan")
    t = T/Tc; h = k*H
    gamma_sq = (alpha*h)**2 - (lam_so**2)/4.0
    gamma = np.sqrt(gamma_sq + 0j)
    if gamma == 0:
        lhs = np.log(max(t,1e-12)) + (psi(0.5 + h/(2.0*t)) - psi(0.5))
        return float(np.real(lhs))
    Aplus  = 0.5 + 1j*lam_so/(4.0*gamma)
    Aminus = 0.5 - 1j*lam_so/(4.0*gamma)
    zplus  = 0.5 + (h + 0.5*lam_so + 1j*gamma)/(2.0*t)
    zminus = 0.5 + (h + 0.5*lam_so - 1j*gamma)/(2.0*t)
    lhs = np.log(max(t,1e-12)) + Aplus*psi(zplus) + Aminus*psi(zminus) - psi(0.5)
    return float(np.real(lhs))

def F1b_orbital(T,H,Tc,k): return F1b_orbital_lhs(T/Tc, k*H)
def F1b_pauli(T,H,Tc,k,alpha,lam_so): return F1b_pauli_so_lhs(T,H,Tc,k,alpha,lam_so)

# ---- Two-band (Gurevich coefficients)

def a_coeffs_from_lams(l11, l22, l12, l21):
    w  = l11*l22 - l12*l21
    lm = l11 - l22
    l0 = np.sqrt(lm*lm + 4.0*l12*l21)
    if l0 < 1e-12: l0 = 1e-12
    a0 = 2.0*w / l0
    a1 = 1.0 + lm / l0
    a2 = 1.0 - lm / l0
    return a0, a1, a2

def U_orb(h_over_t): return psi(0.5 + h_over_t) - psi(0.5)

def F2b_orbital(T, H, Tc, k1, eta, l11, l22, l12, l21):
    if T <= 0 or Tc <= 0 or k1 <= 0 or eta < 0: return float("nan")
    t = T/Tc; Lnt = np.log(max(t,1e-12))
    h1 = k1*H; h2 = eta*h1
    U1 = U_orb(h1/(2.0*t)); U2 = U_orb(h2/(2.0*t))
    a0,a1,a2 = a_coeffs_from_lams(l11,l22,l12,l21)
    return a0*(Lnt+U1)*(Lnt+U2) + a1*(Lnt+U1) + a2*(Lnt+U2)

def U_pauli_band(h, t, alpha, lam_so):
    if t <= 0: return np.nan
    gamma_sq = (alpha*h)**2 - (lam_so**2)/4.0
    gamma = np.sqrt(gamma_sq + 0j)
    if gamma == 0:
        return psi(0.5 + h/(2.0*t)) - psi(0.5)
    Aplus  = 0.5 + 1j*lam_so/(4.0*gamma)
    Aminus = 0.5 - 1j*lam_so/(4.0*gamma)
    zplus  = 0.5 + (h + 0.5*lam_so + 1j*gamma)/(2.0*t)
    zminus = 0.5 + (h + 0.5*lam_so - 1j*gamma)/(2.0*t)
    val = Aplus*psi(zplus) + Aminus*psi(zminus) - psi(0.5)
    return float(np.real(val))

def F2b_pauli(T, H, Tc, k1, eta, l11, l22, l12, l21, a1, a2, s1, s2):
    if T <= 0 or Tc <= 0 or k1 <= 0 or eta < 0: return float("nan")
    t = T/Tc; Lnt = np.log(max(t,1e-12))
    h1 = k1*H; h2 = eta*h1
    P1 = U_pauli_band(h1, t, a1, s1); P2 = U_pauli_band(h2, t, a2, s2)
    A0,A1,A2 = a_coeffs_from_lams(l11,l22,l12,l21)
    return A0*(Lnt+P1)*(Lnt+P2) + A1*(Lnt+P1) + A2*(Lnt+P2)

# ============================== models ==============================

def model_implicit(T_arr, root_fun, args, Hmax=None):
    out, H_hint = [], None
    Ts = np.sort(np.array(T_arr, float))
    Tc = args[0]
    for Ti in Ts:
        root = 0.0 if Ti >= Tc else largest_root(Ti, root_fun, args, H_hint, Hmax)
        out.append(root)
        if not np.isnan(root): H_hint = root
    order = np.argsort(T_arr); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    return np.array(out)[inv]

# ============================ residuals ============================

def penalty_monotone(T, Hpred, weight=2e3, mono_tol=1e-6):
    idx = np.argsort(T)
    d = np.diff(Hpred[idx]) + mono_tol
    return weight * np.maximum(d, 0.0)

def res_1b_orbital(theta, T, H, Tc_fixed=None, Hmax=None):
    if Tc_fixed is None: Tc,k = theta
    else: (k,) = theta; Tc = Tc_fixed
    Hpred = model_implicit(T, F1b_orbital, (Tc,k), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred)])

def res_1b_pauli(theta, T, H, Tc_fixed=None, alpha_fixed=None, lam_fixed=None, Hmax=None):
    q=0
    if Tc_fixed is None: Tc=theta[q]; q+=1
    else: Tc=Tc_fixed
    k=theta[q]; q+=1
    if alpha_fixed is None: alpha=theta[q]; q+=1
    else: alpha=alpha_fixed
    if lam_fixed is None: lam=theta[q]; q+=1
    else: lam=lam_fixed
    Hpred = model_implicit(T, F1b_pauli, (Tc,k,alpha,lam), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred)])

def res_2b_orbital_sym(theta, T, H, Tc_fixed=None, eta_fixed=None,
                       lam11_fixed=None, lam22_fixed=None, lam12_fixed=None,
                       lam_bounds=(0,3), Hmax=None):
    q=0
    if Tc_fixed is None: Tc=theta[q]; q+=1
    else: Tc=Tc_fixed
    k1=theta[q]; q+=1
    if eta_fixed is None: eta=theta[q]; q+=1
    else: eta=eta_fixed
    lam11 = lam11_fixed if lam11_fixed is not None else theta[q]; q += (0 if lam11_fixed is not None else 1)
    lam22 = lam22_fixed if lam22_fixed is not None else theta[q]; q += (0 if lam22_fixed is not None else 1)
    lam12 = lam12_fixed if lam12_fixed is not None else theta[q]; q += (0 if lam12_fixed is not None else 1)
    lam21 = lam12
    lmin,lmax = lam_bounds
    penalty = 0.0
    for x in (lam11, lam22, lam12, lam21):
        if x < lmin or x > lmax:
            penalty += 1e3 * min(abs(x-lmin), abs(x-lmax))**2
    Hpred = model_implicit(T, F2b_orbital, (Tc,k1,eta,lam11,lam22,lam12,lam21), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred), np.array([penalty])])

def res_2b_orbital_untied(theta, T, H, Tc_fixed=None, eta_fixed=None,
                          lam11_fixed=None, lam22_fixed=None, lam12_fixed=None, lam21_fixed=None,
                          lam_bounds=(0,3), Hmax=None):
    q=0
    if Tc_fixed is None: Tc=theta[q]; q+=1
    else: Tc=Tc_fixed
    k1=theta[q]; q+=1
    if eta_fixed is None: eta=theta[q]; q+=1
    else: eta=eta_fixed
    lam11 = lam11_fixed if lam11_fixed is not None else theta[q]; q += (0 if lam11_fixed is not None else 1)
    lam22 = lam22_fixed if lam22_fixed is not None else theta[q]; q += (0 if lam22_fixed is not None else 1)
    lam12 = lam12_fixed if lam12_fixed is not None else theta[q]; q += (0 if lam12_fixed is not None else 1)
    lam21 = lam21_fixed if lam21_fixed is not None else theta[q]; q += (0 if lam21_fixed is not None else 1)
    lmin,lmax = lam_bounds
    penalty = 0.0
    for x in (lam11, lam22, lam12, lam21):
        if x < lmin or x > lmax:
            penalty += 1e3 * min(abs(x-lmin), abs(x-lmax))**2
    Hpred = model_implicit(T, F2b_orbital, (Tc,k1,eta,lam11,lam22,lam12,lam21), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred), np.array([penalty])])

def res_2b_pauli_sym(theta, T, H, Tc_fixed=None, eta_fixed=None,
                     lam11_fixed=None, lam22_fixed=None, lam12_fixed=None,
                     a1_fixed=None, a2_fixed=None, s1_fixed=None, s2_fixed=None,
                     lam_bounds=(0,3), Hmax=None):
    q=0
    if Tc_fixed is None: Tc=theta[q]; q+=1
    else: Tc=Tc_fixed
    k1=theta[q]; q+=1
    if eta_fixed is None: eta=theta[q]; q+=1
    else: eta=eta_fixed
    lam11 = lam11_fixed if lam11_fixed is not None else theta[q]; q += (0 if lam11_fixed is not None else 1)
    lam22 = lam22_fixed if lam22_fixed is not None else theta[q]; q += (0 if lam22_fixed is not None else 1)
    lam12 = lam12_fixed if lam12_fixed is not None else theta[q]; q += (0 if lam12_fixed is not None else 1)
    lam21 = lam12
    a1 = a1_fixed if a1_fixed is not None else theta[q]; q += (0 if a1_fixed is not None else 1)
    a2 = a2_fixed if a2_fixed is not None else theta[q]; q += (0 if a2_fixed is not None else 1)
    s1 = s1_fixed if s1_fixed is not None else theta[q]; q += (0 if s1_fixed is not None else 1)
    s2 = s2_fixed if s2_fixed is not None else theta[q]; q += (0 if s2_fixed is not None else 1)
    lmin,lmax = lam_bounds
    penalty = 0.0
    for x in (lam11, lam22, lam12, lam21):
        if x < lmin or x > lmax:
            penalty += 1e3 * min(abs(x-lmin), abs(x-lmax))**2
    Hpred = model_implicit(T, F2b_pauli, (Tc,k1,eta,lam11,lam22,lam12,lam21,a1,a2,s1,s2), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred), np.array([penalty])])

def res_2b_pauli(theta, T, H, Tc_fixed=None, eta_fixed=None,
                 lam11_fixed=None, lam22_fixed=None, lam12_fixed=None, lam21_fixed=None,
                 a1_fixed=None, a2_fixed=None, s1_fixed=None, s2_fixed=None,
                 lam_bounds=(0,3), Hmax=None):
    q=0
    if Tc_fixed is None: Tc=theta[q]; q+=1
    else: Tc=Tc_fixed
    k1=theta[q]; q+=1
    if eta_fixed is None: eta=theta[q]; q+=1
    else: eta=eta_fixed
    lam11 = lam11_fixed if lam11_fixed is not None else theta[q]; q += (0 if lam11_fixed is not None else 1)
    lam22 = lam22_fixed if lam22_fixed is not None else theta[q]; q += (0 if lam22_fixed is not None else 1)
    lam12 = lam12_fixed if lam12_fixed is not None else theta[q]; q += (0 if lam12_fixed is not None else 1)
    lam21 = lam21_fixed if lam21_fixed is not None else theta[q]; q += (0 if lam21_fixed is not None else 1)
    a1 = a1_fixed if a1_fixed is not None else theta[q]; q += (0 if a1_fixed is not None else 1)
    a2 = a2_fixed if a2_fixed is not None else theta[q]; q += (0 if a2_fixed is not None else 1)
    s1 = s1_fixed if s1_fixed is not None else theta[q]; q += (0 if s1_fixed is not None else 1)
    s2 = s2_fixed if s2_fixed is not None else theta[q]; q += (0 if s2_fixed is not None else 1)
    lmin,lmax = lam_bounds
    penalty = 0.0
    for x in (lam11, lam22, lam12, lam21):
        if x < lmin or x > lmax:
            penalty += 1e3 * min(abs(x-lmin), abs(x-lmax))**2
    Hpred = model_implicit(T, F2b_pauli, (Tc,k1,eta,lam11,lam22,lam12,lam21,a1,a2,s1,s2), Hmax)
    if np.any(np.isnan(Hpred)): return 1e6*np.ones(len(H)+len(H)-1)
    return np.concatenate([Hpred - H, penalty_monotone(T,Hpred), np.array([penalty])])

# ============================== fitters ==============================

def fit_1b_orbital(T, H, Tc_fixed=None):
    Tmax=float(np.max(T)); Hmax=float(np.max(H))
    if Tc_fixed is None:
        p0=[max(Tmax+0.2,2.5), 0.02]; lb=[Tmax+1e-3, 1e-6]; ub=[10.0, 1.0]
        res=least_squares(lambda th:res_1b_orbital(th,T,H,None,Hmax), p0, bounds=(lb,ub),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=600)
        Tc,k=res.x
    else:
        p0=[0.02]; lb=[1e-6]; ub=[1.0]
        res=least_squares(lambda th:res_1b_orbital(th,T,H,Tc_fixed,Hmax), p0, bounds=(lb,ub),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=600)
        k=res.x[0]; Tc=Tc_fixed
    Hpred=model_implicit(T,F1b_orbital,(Tc,k),Hmax)
    return (Tc,k), Hpred, res

def fit_1b_pauli(T, H, Tc_fixed=None, alpha_fixed=None, lam_fixed=None,
                 a_bounds=(0,100), s_bounds=(0,100)):
    Tmax=float(np.max(T)); Hmax=float(np.max(H))
    theta0=[]; lb=[]; ub=[]
    if Tc_fixed is None: theta0+=[max(Tmax+0.2,2.5)]; lb+=[Tmax+1e-3]; ub+=[10.0]
    theta0+=[0.02]; lb+=[1e-6]; ub+=[1.0]
    if alpha_fixed is None: theta0+=[0.7]; lb+=[a_bounds[0]]; ub+=[a_bounds[1]]
    if lam_fixed   is None: theta0+=[0.5]; lb+=[s_bounds[0]]; ub+=[s_bounds[1]]
    res=least_squares(lambda th:res_1b_pauli(th,T,H,Tc_fixed,alpha_fixed,lam_fixed,Hmax),
                      np.array(theta0,float), bounds=(np.array(lb),np.array(ub)),
                      xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=800)
    q=0
    if Tc_fixed is None: Tc=res.x[q]; q+=1
    else: Tc=Tc_fixed
    k=res.x[q]; q+=1
    if alpha_fixed is None: alpha=res.x[q]; q+=1
    else: alpha=alpha_fixed
    if lam_fixed is None: lam=res.x[q]; q+=1
    else: lam=lam_fixed
    Hpred=model_implicit(T,F1b_pauli,(Tc,k,alpha,lam),Hmax)
    return (Tc,k,alpha,lam), Hpred, res

def fit_2b_orbital(T,H, Tc_fixed=None, eta_fixed=None,
                   lam11_fixed=None, lam22_fixed=None, lam12_fixed=None,
                   lam_bounds=(0,3), untie=False, lam21_fixed=None):
    Tmax=float(np.max(T)); Hmax=float(np.max(H))
    theta0=[]; lb=[]; ub=[]
    if Tc_fixed is None: theta0+=[max(Tmax+0.2,2.5)]; lb+=[Tmax+1e-3]; ub+=[10.0]
    theta0+=[0.02]; lb+=[1e-6]; ub+=[1.0]
    if eta_fixed is None: theta0+=[0.3]; lb+=[1e-4]; ub+=[100.0]
    lmin,lmax = lam_bounds
    if lam11_fixed is None: theta0+=[0.8]; lb+=[lmin]; ub+=[lmax]
    if lam22_fixed is None: theta0+=[0.3]; lb+=[lmin]; ub+=[lmax]
    if lam12_fixed is None: theta0+=[0.1]; lb+=[lmin]; ub+=[lmax]
    if untie and (lam21_fixed is None): theta0+=[0.1]; lb+=[lmin]; ub+=[lmax]

    if not untie:
        res=least_squares(lambda th:res_2b_orbital_sym(th,T,H,Tc_fixed,eta_fixed,lam11_fixed,lam22_fixed,lam12_fixed,lam_bounds,Hmax),
                          np.array(theta0,float), bounds=(np.array(lb),np.array(ub)),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=1500)
        q=0
        if Tc_fixed is None: Tc=res.x[q]; q+=1
        else: Tc=Tc_fixed
        k1=res.x[q]; q+=1
        if eta_fixed is None: eta=res.x[q]; q+=1
        else: eta=eta_fixed
        lam11 = lam11_fixed if lam11_fixed is not None else res.x[q]; q += (0 if lam11_fixed is not None else 1)
        lam22 = lam22_fixed if lam22_fixed is not None else res.x[q]; q += (0 if lam22_fixed is not None else 1)
        lam12 = lam12_fixed if lam12_fixed is not None else res.x[q]; q += (0 if lam12_fixed is not None else 1)
        lam21 = lam12
    else:
        res=least_squares(lambda th:res_2b_orbital_untied(th,T,H,Tc_fixed,eta_fixed,lam11_fixed,lam22_fixed,lam12_fixed,lam21_fixed,lam_bounds,Hmax),
                          np.array(theta0,float), bounds=(np.array(lb),np.array(ub)),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=1800)
        q=0
        if Tc_fixed is None: Tc=res.x[q]; q+=1
        else: Tc=Tc_fixed
        k1=res.x[q]; q+=1
        if eta_fixed is None: eta=res.x[q]; q+=1
        else: eta=eta_fixed
        lam11 = lam11_fixed if lam11_fixed is not None else res.x[q]; q += (0 if lam11_fixed is not None else 1)
        lam22 = lam22_fixed if lam22_fixed is not None else res.x[q]; q += (0 if lam22_fixed is not None else 1)
        lam12 = lam12_fixed if lam12_fixed is not None else res.x[q]; q += (0 if lam12_fixed is not None else 1)
        lam21 = lam21_fixed if lam21_fixed is not None else res.x[q]; q += (0 if lam21_fixed is not None else 1)

    Hpred=model_implicit(T, F2b_orbital, (Tc,k1,eta,lam11,lam22,lam12,lam21), Hmax)
    return (Tc,k1,eta,lam11,lam22,lam12,lam21), Hpred, res

def fit_2b_pauli(T,H, Tc_fixed=None, eta_fixed=None,
                 lam11_fixed=None, lam22_fixed=None, lam12_fixed=None, lam21_fixed=None,
                 a1_fixed=None,a2_fixed=None,s1_fixed=None,s2_fixed=None,
                 lam_bounds=(0,3), a1_bounds=(0,100), a2_bounds=(0,100),
                 s1_bounds=(0,100), s2_bounds=(0,100), untie=False):
    Tmax=float(np.max(T)); Hmax=float(np.max(H))
    theta0=[]; lb=[]; ub=[]
    if Tc_fixed is None: theta0+=[max(Tmax+0.2,2.5)]; lb+=[Tmax+1e-3]; ub+=[10.0]
    theta0+=[0.02]; lb+=[1e-6]; ub+=[1.0]
    if eta_fixed is None: theta0+=[0.3]; lb+=[1e-4]; ub+=[100.0]
    lmin,lmax = lam_bounds
    if lam11_fixed is None: theta0+=[0.8]; lb+=[lmin]; ub+=[lmax]
    if lam22_fixed is None: theta0+=[0.3]; lb+=[lmin]; ub+=[lmax]
    if lam12_fixed is None: theta0+=[0.1]; lb+=[lmin]; ub+=[lmax]
    if untie and (lam21_fixed is None): theta0+=[0.1]; lb+=[lmin]; ub+=[lmax]
    if a1_fixed is None: theta0+=[0.7]; lb+=[a1_bounds[0]]; ub+=[a1_bounds[1]]
    if a2_fixed is None: theta0+=[0.7]; lb+=[a2_bounds[0]]; ub+=[a2_bounds[1]]
    if s1_fixed is None: theta0+=[0.5]; lb+=[s1_bounds[0]]; ub+=[s1_bounds[1]]
    if s2_fixed is None: theta0+=[0.5]; lb+=[s2_bounds[0]]; ub+=[s2_bounds[1]]

    theta0 = np.array(theta0,float)
    lb = np.array(lb,float); ub = np.array(ub,float)

    if not untie:
        res=least_squares(lambda th:res_2b_pauli_sym(th,T,H,Tc_fixed,eta_fixed,
                                                     lam11_fixed,lam22_fixed,lam12_fixed,
                                                     a1_fixed,a2_fixed,s1_fixed,s2_fixed,
                                                     lam_bounds,Hmax),
                          theta0, bounds=(lb,ub),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=2500)
        q=0
        if Tc_fixed is None: Tc=res.x[q]; q+=1
        else: Tc=Tc_fixed
        k1=res.x[q]; q+=1
        if eta_fixed is None: eta=res.x[q]; q+=1
        else: eta=eta_fixed
        lam11 = lam11_fixed if lam11_fixed is not None else res.x[q]; q += (0 if lam11_fixed is not None else 1)
        lam22 = lam22_fixed if lam22_fixed is not None else res.x[q]; q += (0 if lam22_fixed is not None else 1)
        lam12 = lam12_fixed if lam12_fixed is not None else res.x[q]; q += (0 if lam12_fixed is not None else 1)
        lam21 = lam12
        a1 = a1_fixed if a1_fixed is not None else res.x[q]; q += (0 if a1_fixed is not None else 1)
        a2 = a2_fixed if a2_fixed is not None else res.x[q]; q += (0 if a2_fixed is not None else 1)
        s1 = s1_fixed if s1_fixed is not None else res.x[q]; q += (0 if s1_fixed is not None else 1)
        s2 = s2_fixed if s2_fixed is not None else res.x[q]; q += (0 if s2_fixed is not None else 1)
    else:
        res=least_squares(lambda th:res_2b_pauli(th,T,H,Tc_fixed,eta_fixed,
                                                 lam11_fixed,lam22_fixed,lam12_fixed,lam21_fixed,
                                                 a1_fixed,a2_fixed,s1_fixed,s2_fixed,
                                                 lam_bounds,Hmax),
                          theta0, bounds=(lb,ub),
                          xtol=1e-6, ftol=1e-6, gtol=1e-6, max_nfev=2500)
        q=0
        if Tc_fixed is None: Tc=res.x[q]; q+=1
        else: Tc=Tc_fixed
        k1=res.x[q]; q+=1
        if eta_fixed is None: eta=res.x[q]; q+=1
        else: eta=eta_fixed
        lam11 = lam11_fixed if lam11_fixed is not None else res.x[q]; q += (0 if lam11_fixed is not None else 1)
        lam22 = lam22_fixed if lam22_fixed is not None else res.x[q]; q += (0 if lam22_fixed is not None else 1)
        lam12 = lam12_fixed if lam12_fixed is not None else res.x[q]; q += (0 if lam12_fixed is not None else 1)
        lam21 = lam21_fixed if lam21_fixed is not None else res.x[q]; q += (0 if lam21_fixed is not None else 1)
        a1 = a1_fixed if a1_fixed is not None else res.x[q]; q += (0 if a1_fixed is not None else 1)
        a2 = a2_fixed if a2_fixed is not None else res.x[q]; q += (0 if a2_fixed is not None else 1)
        s1 = s1_fixed if s1_fixed is not None else res.x[q]; q += (0 if s1_fixed is not None else 1)
        s2 = s2_fixed if s2_fixed is not None else res.x[q]; q += (0 if s2_fixed is not None else 1)

    Hpred=model_implicit(T, F2b_pauli, (Tc,k1,eta,lam11,lam22,lam12,lam21,a1,a2,s1,s2), Hmax)
    return (Tc,k1,eta,lam11,lam22,lam12,lam21,a1,a2,s1,s2), Hpred, res

# ============================ CLI parsing ============================

def parse_args(argv):
    if len(argv) < 3:
        print("Usage: python hc2_whh_fit.py <files> <mode> "
              "(mode = orbital | pauli | 2band_orbital | 2band_pauli) "
              "[--Tc_fixed=K] "
              "[--alpha=.. --lambda_so=.. --alpha_bounds=lo,hi --lambda_bounds=lo,hi] "
              "[--eta=..] [--lam_bounds=lo,hi] [--untie_interband] "
              "[--lam11=.. --lam22=.. --lam12=.. --lam21=..] "
              "[--alpha1=.. --alpha2=.. --lambda_so1=.. --lambda_so2=..] "
              "[--alpha1_bounds=lo,hi --alpha2_bounds=lo,hi "
              "--lambda_so1_bounds=lo,hi --lambda_so2_bounds=lo,hi] "
              "[--no_stage2_fix_lams]")
        sys.exit(1)

    files_arg = argv[1]
    mode = argv[2].strip().lower()
    files = parse_files_arg(files_arg)

    # One-band defaults (fit)
    Tc_fixed = None
    alpha = None
    lam = None
    alpha_bounds = (0.0, 100.0)
    lambda_bounds = (0.0, 100.0)

    # Two-band shared
    eta = None
    lam_bounds = (0.0, 3.0)
    untie_interband = False

    # Two-band couplings (optional fixed)
    lam11 = None; lam22 = None; lam12 = None; lam21 = None

    # Two-band Pauli+SO (fit)
    alpha1=None; alpha2=None; lamso1=None; lamso2=None
    alpha1_bounds=(0.0,100.0); alpha2_bounds=(0.0,100.0)
    lamso1_bounds=(0.0,100.0); lamso2_bounds=(0.0,100.0)

    # Hardened 2-stage control (for 2band_pauli)
    stage2_fix_lams = True  # default: fix λ_ij from stage1 when going to Pauli

    for arg in argv[3:]:
        if not arg.startswith("--"): continue
        key, val = arg.split("=",1) if "=" in arg else (arg, None)
        if key == "--Tc_fixed": Tc_fixed = float(val)
        elif key == "--alpha": alpha = float(val)
        elif key == "--lambda_so": lam = float(val)
        elif key == "--alpha_bounds": lo,hi = val.split(","); alpha_bounds=(float(lo),float(hi))
        elif key == "--lambda_bounds": lo,hi = val.split(","); lambda_bounds=(float(lo),float(hi))
        elif key == "--eta": eta = float(val)
        elif key == "--lam_bounds": lo,hi = val.split(","); lam_bounds=(float(lo),float(hi))
        elif key == "--untie_interband": untie_interband = True
        elif key == "--lam11": lam11 = float(val)
        elif key == "--lam22": lam22 = float(val)
        elif key == "--lam12": lam12 = float(val)
        elif key == "--lam21": lam21 = float(val)
        elif key == "--alpha1": alpha1 = float(val)
        elif key == "--alpha2": alpha2 = float(val)
        elif key == "--lambda_so1": lamso1 = float(val)
        elif key == "--lambda_so2": lamso2 = float(val)
        elif key == "--alpha1_bounds": lo,hi = val.split(","); alpha1_bounds=(float(lo),float(hi))
        elif key == "--alpha2_bounds": lo,hi = val.split(","); alpha2_bounds=(float(lo),float(hi))
        elif key == "--lambda_so1_bounds": lo,hi = val.split(","); lamso1_bounds=(float(lo),float(hi))
        elif key == "--lambda_so2_bounds": lo,hi = val.split(","); lamso2_bounds=(float(lo),float(hi))
        elif key == "--no_stage2_fix_lams": stage2_fix_lams = False

    if mode not in ("orbital","pauli","2band_orbital","2band_pauli"):
        raise ValueError("Mode must be 'orbital' | 'pauli' | '2band_orbital' | '2band_pauli'")
    if not files:
        raise ValueError("Please provide at least one CSV file (T,H).")

    return (files, mode, Tc_fixed, alpha, lam, alpha_bounds, lambda_bounds,
            eta, lam_bounds, untie_interband,
            lam11, lam22, lam12, lam21,
            alpha1, alpha2, lamso1, lamso2,
            alpha1_bounds, alpha2_bounds, lamso1_bounds, lamso2_bounds,
            stage2_fix_lams)

# ============================== Driver ==============================

def main(argv):
    (files, mode, Tc_fixed, alpha, lam, alpha_bounds, lambda_bounds,
     eta, lam_bounds, untie_interband,
     lam11, lam22, lam12, lam21,
     alpha1, alpha2, lamso1, lamso2,
     alpha1_bounds, alpha2_bounds, lamso1_bounds, lamso2_bounds,
     stage2_fix_lams) = parse_args(argv)

    combined_rows = []
    perfile_params = {}

    def suggest_if_drifting(info):
        # Heuristics: λ_ij stuck at bounds or Jacobian rank small or res.success False
        suggestions = []
        if "res" in info:
            res = info["res"]
            if not getattr(res, "success", True):
                suggestions.append("Optimizer did not fully converge.")
            # rank check if available
            if getattr(res, "active_mask", None) is not None and np.any(res.active_mask != 0):
                suggestions.append("Some parameters hit bounds (active_mask non-zero).")
        for k in ("lam11","lam22","lam12","lam21"):
            v = info.get(k, None)
            if v is not None:
                lo,hi = lam_bounds
                if abs(v-lo) < 1e-6 or abs(v-hi) < 1e-6:
                    suggestions.append(f"{k} is at bound ({v:.3g}).")
        if suggestions:
            msg = ("Fit may be drifting / ill-conditioned: " +
                   "; ".join(suggestions) +
                   ". Try fixing λ_ij (e.g., --lam11=0.8 --lam22=0.3 --lam12=0.1" +
                   ("" if untie_interband else " (lam21 tied to lam12)") +
                   ") and refit; or widen --lam_bounds.")
            print(" [suggestion]", msg)

    for path in files:
        df = pd.read_csv(path, header=None, names=["T","H"])
        if not np.issubdtype(df["T"].dtype, np.number):
            df = pd.read_csv(path); df = df.iloc[:, :2]; df.columns = ["T","H"]
        T = df["T"].to_numpy(float); H = df["H"].to_numpy(float)
        Hmax_data = float(np.max(H))
        tag = Path(path).stem

        if mode == "orbital":
            (Tc, k), Hpred, res = fit_1b_orbital(T, H, Tc_fixed=Tc_fixed)
            base = f"{tag}_orbital"
            params_out = {"Tc_K":Tc, "k":k}
        elif mode == "pauli":
            (Tc, k, alpha_fit, lam_fit), Hpred, res = fit_1b_pauli(
                T, H, Tc_fixed=Tc_fixed, alpha_fixed=alpha, lam_fixed=lam,
                a_bounds=alpha_bounds, s_bounds=lambda_bounds
            )
            base = f"{tag}_pauli"
            params_out = {"Tc_K":Tc, "k":k, "alpha":alpha_fit, "lambda_so":lam_fit}
        elif mode == "2band_orbital":
            (Tc, k1, eta_fit, L11, L22, L12, L21), Hpred, res = fit_2b_orbital(
                T, H, Tc_fixed=Tc_fixed, eta_fixed=eta,
                lam11_fixed=lam11, lam22_fixed=lam22, lam12_fixed=lam12,
                lam_bounds=lam_bounds, untie=untie_interband, lam21_fixed=lam21
            )
            base = f"{tag}_2band_orbital"
            params_out = {"Tc_K":Tc, "k1":k1, "eta":eta_fit,
                          "lam11":L11, "lam22":L22, "lam12":L12, "lam21":L21}
            suggest_if_drifting({"res":res, "lam11":L11, "lam22":L22, "lam12":L12, "lam21":L21})
        elif mode == "2band_pauli":
            # Hardened two-stage: Stage 1 seed (2band_orbital)
            print(f"\n[{tag}] Two-stage stabilization: fitting two-band orbital for seeds...")
            (Tc1, k1_1, eta1, L11, L22, L12, L21), _, res1 = fit_2b_orbital(
                T, H, Tc_fixed=Tc_fixed, eta_fixed=eta,
                lam11_fixed=lam11, lam22_fixed=lam22, lam12_fixed=lam12,
                lam_bounds=lam_bounds, untie=untie_interband, lam21_fixed=lam21
            )
            suggest_if_drifting({"res":res1, "lam11":L11, "lam22":L22, "lam12":L12, "lam21":L21})

            # Stage 2 Pauli: by default FIX λ_ij from stage 1 for stability
            print(f"[{tag}] Stage 2: two-band Pauli fit "
                  f"({'fixing' if stage2_fix_lams else 'fitting'} λ_ij from stage1).")
            lam11_s = L11 if stage2_fix_lams else lam11
            lam22_s = L22 if stage2_fix_lams else lam22
            lam12_s = L12 if stage2_fix_lams else lam12
            lam21_s = L21 if stage2_fix_lams else (L12 if not untie_interband else lam21)

            # Prefer Tc from stage1 unless Tc_fixed provided
            Tc_seed = Tc_fixed if Tc_fixed is not None else Tc1
            (Tc, k1, eta_fit, L11p, L22p, L12p, L21p, a1,a2,s1,s2), Hpred, res = fit_2b_pauli(
                T, H, Tc_fixed=Tc_seed, eta_fixed=eta1,
                lam11_fixed=lam11_s, lam22_fixed=lam22_s, lam12_fixed=lam12_s, lam21_fixed=lam21_s,
                a1_fixed=alpha1, a2_fixed=alpha2, s1_fixed=lamso1, s2_fixed=lamso2,
                lam_bounds=lam_bounds, a1_bounds=alpha1_bounds, a2_bounds=alpha2_bounds,
                s1_bounds=lamso1_bounds, s2_bounds=lamso2_bounds, untie=untie_interband
            )
            base = f"{tag}_2band_pauli"
            params_out = {"Tc_K":Tc, "k1":k1, "eta":eta_fit,
                          "lam11":L11p, "lam22":L22p, "lam12":L12p, "lam21":L21p,
                          "alpha1":a1, "alpha2":a2, "lambda_so1":s1, "lambda_so2":s2}
            suggest_if_drifting({"res":res, "lam11":L11p, "lam22":L22p, "lam12":L12p, "lam21":L21p})
        else:
            raise ValueError("Unknown mode")

        # Monotone at the data points
        Hfit = enforce_strict_monotone(T, Hpred)

        # Metrics
        resid = Hfit - H
        rmse = float(np.sqrt(np.mean(resid**2)))
        mae  = float(np.mean(np.abs(resid)))
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((H - np.mean(H))**2))
        R2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")
        strict_flag = bool(np.all(np.diff(Hfit[np.argsort(T)]) < 0.0))

        # Hc2(0) approximate via small T
        T0 = 1e-3
        if mode == "orbital":
            Hc2_0 = largest_root(T0, F1b_orbital, (Tc, params_out["k"]), H_data_max=Hmax_data)
        elif mode == "pauli":
            Hc2_0 = largest_root(T0, F1b_pauli, (Tc, params_out["k"], params_out["alpha"], params_out["lambda_so"]),
                                 H_data_max=Hmax_data)
        elif mode == "2band_orbital":
            Hc2_0 = largest_root(T0, F2b_orbital, (Tc, params_out["k1"], params_out["eta"],
                                                   params_out["lam11"], params_out["lam22"], params_out["lam12"], params_out["lam21"]),
                                                   H_data_max=Hmax_data)
        else:
            Hc2_0 = largest_root(T0, F2b_pauli, (Tc, params_out["k1"], params_out["eta"],
                                                 params_out["lam11"], params_out["lam22"], params_out["lam12"], params_out["lam21"],
                                                 params_out["alpha1"], params_out["alpha2"], params_out["lambda_so1"], params_out["lambda_so2"]),
                                                 H_data_max=Hmax_data)

        # Save per-file outputs
        pd.DataFrame({"T_K": T, "Hc2_T_data": H, "Hc2_T_fit_mono": Hfit}).to_csv(f"{base}_results.csv", index=False)

        # Params CSV row
        row = {"file": path, "mode": mode, "Tc_K": params_out["Tc_K"],
               "RMSE_T": rmse, "MAE_T": mae, "R2": R2,
               "Hc2_approx_0K_T0=0.001K": Hc2_0,
               "strictly_monotone_decreasing": strict_flag}
        if mode == "orbital":
            row["k_scale(h=kH)"] = params_out["k"]
        elif mode == "pauli":
            row.update({"k_scale(h=kH)": params_out["k"],
                        "alpha_Maki": params_out["alpha"], "lambda_so": params_out["lambda_so"],
                        "alpha_bounds": str(alpha_bounds), "lambda_bounds": str(lambda_bounds)})
        else:
            row.update({"k1_scale(h1=k1*H)": params_out["k1"], "eta=D2/D1": params_out["eta"],
                        "lam11": params_out.get("lam11"), "lam22": params_out.get("lam22"),
                        "lam12": params_out.get("lam12"), "lam21": params_out.get("lam21")})
            if mode == "2band_pauli":
                row.update({"alpha1_Maki": params_out["alpha1"], "alpha2_Maki": params_out["alpha2"],
                            "lambda_so1": params_out["lambda_so1"], "lambda_so2": params_out["lambda_so2"],
                            "alpha1_bounds": str(alpha1_bounds), "alpha2_bounds": str(alpha2_bounds),
                            "lambda_so1_bounds": str(lamso1_bounds), "lambda_so2_bounds": str(lamso2_bounds)})

        pd.DataFrame([row]).to_csv(f"{base}_params.csv", index=False)

        # Console summary
        print("\n=== Fit summary:", tag, "===")
        print(f" Mode: {mode}")
        print(f" Tc   : {params_out['Tc_K']:.6g} K")
        if mode in ("orbital","pauli"):
            print(f" k    : {row.get('k_scale(h=kH)', params_out.get('k')):.6g}  (h = k*H)")
        else:
            print(f" k1   : {params_out['k1']:.6g}  (h1 = k1*H),  eta: {params_out['eta']:.6g}")
            if 'lam11' in params_out:
                print(f" lambdas: lam11={params_out['lam11']:.4g}, lam22={params_out['lam22']:.4g}, "
                      f"lam12={params_out['lam12']:.4g}, lam21={params_out['lam21']:.4g}")
        if mode == "pauli":
            print(f" alpha: {params_out['alpha']:.6g}  lambda_so: {params_out['lambda_so']:.6g}")
        if mode == "2band_pauli":
            print(f" alpha1: {params_out['alpha1']:.6g}  alpha2: {params_out['alpha2']:.6g}  "
                  f"lambda_so1: {params_out['lambda_so1']:.6g}  lambda_so2: {params_out['lambda_so2']:.6g}")
        print(f" R^2  : {R2:.6f}")
        print(f" Hc2_max (≈0 K at T=0.001 K): {Hc2_0:.6g} T")
        print(f" Strictly monotone: {strict_flag}")
        print(" Files:", f"{base}_results.csv,", f"{base}_params.csv")

        perfile_params[path] = (mode, params_out)
        combined_rows.append(row)

    # Combined params table
    comb = pd.DataFrame(combined_rows)
    comb_name = f"combined_{mode}_params.csv"
    comb.to_csv(comb_name, index=False)

    # Combined plot (0 → Tc)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colors = plt.cm.tab10.colors
    for i, path in enumerate(files):
        tag = Path(path).stem
        res_csv = f"{tag}_{mode}_results.csv"
        if not Path(res_csv).exists(): continue
        df_res = pd.read_csv(res_csv)
        order = np.argsort(df_res["T_K"].to_numpy())
        ax.scatter(df_res["T_K"].to_numpy()[order],
                   df_res["Hc2_T_data"].to_numpy()[order],
                   label=f"{tag} data", color=colors[i % 10], s=22)

        mode_i, p = perfile_params[path]
        Tc = p["Tc_K"]
        T_eps = max(1e-6, Tc/2000.0); T_plot = np.linspace(T_eps, Tc, 500)

        if mode == "orbital":
            H_plot = model_implicit(T_plot, F1b_orbital, (Tc, p["k"]))
        elif mode == "pauli":
            H_plot = model_implicit(T_plot, F1b_pauli, (Tc, p["k"], p["alpha"], p["lambda_so"]))
        elif mode == "2band_orbital":
            H_plot = model_implicit(T_plot, F2b_orbital,
                                    (Tc, p["k1"], p["eta"], p["lam11"], p["lam22"], p["lam12"], p["lam21"]))
        else:
            H_plot = model_implicit(T_plot, F2b_pauli,
                                    (Tc, p["k1"], p["eta"], p["lam11"], p["lam22"], p["lam12"], p["lam21"],
                                     p["alpha1"], p["alpha2"], p["lambda_so1"], p["lambda_so2"]))
        H_plot = enforce_strict_monotone(T_plot, H_plot)
        T_full = np.concatenate(([0.0], T_plot))
        H_full = np.concatenate(([H_plot[0]], H_plot))
        ax.plot(T_full, H_full, color=colors[i % 10], linestyle="--", label=f"{tag} fit")

    ax.set_xlabel("Temperature T (K)")
    ax.set_ylabel(r"$\mu_0 H_{c2}$ (T)")
    ax.set_title(f"WHH fits ({mode}) — curves from 0 → Tc")
    ax.legend(fontsize=9)
    ax.tick_params(which="both", direction="in", top=True, right=True)
    fig.tight_layout()
    comb_plot_name = f"combined_{mode}_plot"
    plt.savefig(comb_plot_name + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(comb_plot_name + ".pdf", bbox_inches="tight")
    plt.close()

    print("\n=== Combined outputs ===")
    print(" Combined params:", comb_name)
    print(" Combined plot  :", comb_plot_name + ".pdf")

if __name__ == "__main__":
    main(sys.argv)
