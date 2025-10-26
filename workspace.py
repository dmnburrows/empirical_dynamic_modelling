# ===============================================================
# FIXED VERSION — safe for both IPython and Python multiprocessing
# ===============================================================

import multiprocessing, sys, os, time
# Force safe start method on macOS + IPython (spawns clean processes)
multiprocessing.set_start_method('spawn', force=True)

# Register the current module as importable (needed for joblib workers)
sys.modules['workspace'] = sys.modules['__main__']

# ===============================================================
# Imports and setup
# ===============================================================
from brian2 import *
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Callable
import numpy as np
import pandas as pd
import glob
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from IPython.display import HTML
import matplotlib.animation as animation
import matplotlib as mpl
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
mpl.rcParams['animation.embed_limit'] = 1000  # in MB
print(sys.version)

# Custom paths
sys.path.insert(0, "/Users/k2585057/Dropbox/PhD/Analysis/my_scripts/GitHub/gene_snn/")
sys.path.insert(0, "/Users/k2585057/Dropbox/PhD/Analysis/my_scripts/GitHub/neuropix_process/sig/")
sys.path.insert(0, "/Users/k2585057/Dropbox/PhD/Analysis/my_scripts/GitHub/neuropix_process/visualise/")
from model_builder import *
from model_params import *
import visualise as vif
import process as prf
import firing_metrics as fif

# ===============================================================
# Functions
# ===============================================================

def _run(pars):
    b2.start_scope()
    SEED = 42
    b2.seed(SEED); np.random.seed(SEED)
    
    sim_time=60.0*second
    warmup=1.0*second
    
    model = braintype(pars = pars)
    model.DE(); model.neuron(); model.population(); model.region(); model.synapse()
    
    R = Registry()
    R.add_mechanism(model.lif_dyn); R.add_mechanism(model.dirsyn_dyn)
    R.add_neuron(model.exc_nrn); R.add_neuron(model.inh_nrn)
    R.add_population(model.cort_exc_pop); R.add_population(model.cort_inh_pop)
    R.add_region(model.cort_reg)
    R.add_synapse(model.ee_syn); R.add_synapse(model.ei_syn)
    R.add_synapse(model.ie_syn); R.add_synapse(model.ii_syn)
    
    builder = BrianBuilder(R, default_dt=0.1*b2.ms, default_method='heun')
    E = builder.build_population("cortical_excitatory")
    I = builder.build_population("cortical_inhibitory")
    builder.build_synapse("cortical_excitatory_to_cortical_excitatory")
    builder.build_synapse("cortical_excitatory_to_cortical_inhibitory")
    builder.build_synapse("cortical_inhibitory_to_cortical_excitatory")
    builder.build_synapse("cortical_inhibitory_to_cortical_inhibitory")
    
    builder.net.run(warmup)
    spk_E = b2.SpikeMonitor(E); spk_I = b2.SpikeMonitor(I)
    builder.net.add(spk_E, spk_I)
    builder.net.run(sim_time)
    
    rate = 100
    n_time = int(rate * sim_time)
    E_mat, _ = prf.bin_b2(np.array(spk_E.i), np.array(spk_E.t - warmup),
                          rate=rate, N=spk_E.source.N, n_time=n_time)
    I_mat, _ = prf.bin_b2(np.array(spk_I.i), np.array(spk_I.t - warmup),
                          rate=rate, N=spk_I.source.N, n_time=n_time)
    return E_mat, I_mat


def spk_to_fl(S, dt_native, tau_d=0.7, A=1.0, ksat=1.0, F0=0.0, add_noise=False):
    S = np.asarray(S, dtype=np.float32)
    N, T = S.shape
    alpha = np.exp(-dt_native / tau_d)
    C = np.zeros_like(S)
    for t in range(1, T):
        C[:, t] = alpha * C[:, t-1] + (1 - alpha) * S[:, t]
    F = F0 + (A * C) / (1.0 + ksat * C)
    if add_noise:
        F += 0.01 * np.random.randn(*F.shape)
    return F


from sklearn.decomposition import PCA
def participation_ratio(X):
    Xz = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)
    pca = PCA(svd_solver='randomized')
    pca.fit(Xz.T)
    lam = pca.singular_values_**2
    return (lam.sum()**2) / (np.sum(lam**2)), lam


from pyEDM import EmbedDimension
def E_from_pcs(PCs, maxE=20, Tp=1, tau=10, lib_frac=0.6):
    PCs = np.asarray(PCs)
    T, K = PCs.shape
    df = pd.DataFrame(PCs, columns=[f"PC{i+1}" for i in range(K)])
    df.insert(0, "Time", np.arange(T))
    split = int(lib_frac * T)
    lib, pred = f"1 {split}", f"{split+1} {T}"
    E_per_pc, rho_per_pc = np.full(K, np.nan), np.full(K, np.nan)
    for k in range(K):
        target = f"PC{k+1}"
        out = EmbedDimension(dataFrame=df, columns=target, target=target,
                             maxE=maxE, lib=lib, pred=pred, Tp=Tp, tau=tau,
                             embedded=False, numProcess=1, showPlot=False)
        best = out.loc[out['rho'].idxmax()]
        E_per_pc[k], rho_per_pc[k] = best['E'], best['rho']
    return np.mean(E_per_pc), np.mean(rho_per_pc)

# # ===============================================================
# # Parameters (identical to your original)
# # ===============================================================
base_pars = {
    'spike_threshold_u': -50,
    'spike_threshold_sigma': 5,
    'rmp_u': -70,
    'rmp_sigma': 1,
    'synaptic_delay_u': 1.0,
    'synaptic_delay_sigma': 0.15,
    'tau_excitatory_u': 12,
    'tau_inhibitory_u': 6,
    'tau_sigma': 2,
    'refractory_excitatory_u': 2.0,
    'refractory_inhibitory_u': 1.0,
    'refractory_sigma': 0.5,
    'excitatory_size': 1000,
    'inhibitory_size': 400,
    'e-e_connectivity_p': 0.05,
    'e-i_connectivity_p': 0.08,
    'i-e_connectivity_p': 0.25,
    'i-i_connectivity_p': 0.20,
    'e-e_synw_u': 3.5,
    'e-i_synw_u': 1.9,
    'i-e_synw_u': 2.1,
    'i-i_synw_u': 1.7,
    'synw_sigma': 0.3,
    'PI_size': 1000,
    'PI_p': 0.1,
    'PI_rates_u_excitatory': 8,
    'PI_rates_u_inhibitory': 8,
    'PI_rates_sigma': 0.1,
    'PI_weights_sigma': 0.2,
    'PI_weights_u_excitatory': 1.0,
    'PI_weights_u_inhibitory': 1.0
}

# param_grid = {
#     'PI_rates_sigma':      np.linspace(0.1, 5, 20),
#     'synw_sigma':          np.linspace(0.1, 1.0, 10),
#     'e-e_connectivity_p':  np.linspace(0.02, 0.8, 20),
#     'i-e_connectivity_p':  np.linspace(0.1, 0.8, 20),
#     'e-i_connectivity_p':  np.linspace(0.04, 0.8, 20),
#     'i-i_connectivity_p':  np.linspace(0.1, 0.8, 20),
#     'synaptic_delay_u':    np.linspace(0.5, 10.0, 20),
#     'tau_excitatory_u':    np.linspace(6, 20, 10),
#     'tau_inhibitory_u':    np.linspace(3, 12, 10),
#     'e-e_synw_u':          np.linspace(1, 6, 20),
#     'e-i_synw_u':          np.linspace(1, 8, 20),
#     'i-e_synw_u':          np.linspace(1, 8, 20),
#     'i-i_synw_u':          np.linspace(1, 8, 20)
# }

# # ===============================================================
# # Evaluation + grid
# # ===============================================================
# from sklearn.preprocessing import StandardScaler
# def evaluate_param(param_name, value, base_pars):
#     pars = base_pars.copy()
#     pars[param_name] = value
#     try:
#         E_mat, I_mat = _run(pars)
#         mat = np.array(sp.vstack((E_mat, I_mat)).todense())
#         fmat = spk_to_fl(S=mat, dt_native=1/100)
#         PR, _ = participation_ratio(fmat)
#         X = fmat.T[200:]
#         sc = StandardScaler().fit_transform(X)
#         pca = PCA(n_components=10).fit_transform(sc)
#         E_mean, _ = E_from_pcs(pca)
#         return {'param': param_name, 'value': value, 'PR': PR, 'E': E_mean}
#     except Exception as e:
#         print(f"Error on {param_name}={value}: {e}")
#         return {'param': param_name, 'value': value, 'PR': np.nan, 'E': np.nan}


# from joblib import Parallel, delayed
# def run_grid(param_grid, base_pars, n_jobs=4, save_csv='E_PR_grid_results.csv'):
#     tasks = [(p, v) for p, vals in param_grid.items() for v in vals]
#     print(f"Total runs: {len(tasks)}")
#     t0 = time.time()
#     results = Parallel(n_jobs=n_jobs, verbose=10)(
#         delayed(evaluate_param)(p, v, base_pars) for p, v in tasks
#     )
#     df = pd.DataFrame(results)
#     df.to_csv(save_csv, index=False)
#     print(f"\nGrid complete. Saved to {save_csv}")
#     print(f"Elapsed: {(time.time()-t0)/3600:.2f} hours")
#     return df

# # ===============================================================
# # Run
# # ===============================================================
# if __name__ == "__main__":
#     df = run_grid(param_grid, base_pars, n_jobs=4)
#     print(df.head())


# ===============================================================
# Focused 2D grid: e-e_connectivity_p × e-e_synw_u
# ===============================================================
ee_connectivity_vals = np.linspace(0.02, 0.8, 10)
ee_synw_vals         = np.linspace(1, 4, 10)

def evaluate_pair(ee_p, ee_w, base_pars):
    """Run network for one combination of e-e_connectivity_p and e-e_synw_u."""
    pars = base_pars.copy()
    pars['e-e_connectivity_p'] = ee_p
    pars['e-e_synw_u'] = ee_w
    try:
        E_mat, I_mat = _run(pars)
        mat = np.array(sp.vstack((E_mat, I_mat)).todense())
        fmat = spk_to_fl(S=mat, dt_native=1/100)
        PR, _ = participation_ratio(fmat)
        X = fmat.T[200:]
        sc = StandardScaler().fit_transform(X)
        pca = PCA(n_components=10).fit_transform(sc)
        E_mean, _ = E_from_pcs(pca)
        return {
            'e-e_connectivity_p': ee_p,
            'e-e_synw_u': ee_w,
            'PR': PR,
            'E': E_mean
        }
    except Exception as e:
        print(f"Error on p={ee_p}, w={ee_w}: {e}")
        return {
            'e-e_connectivity_p': ee_p,
            'e-e_synw_u': ee_w,
            'PR': np.nan,
            'E': np.nan
        }

# ===============================================================
# Run 2D grid
# ===============================================================
from joblib import Parallel, delayed

def run_2D_grid(ee_connectivity_vals, ee_synw_vals, base_pars, n_jobs=4, save_csv='E_PR_2D_grid.csv'):
    tasks = [(p, w) for p in ee_connectivity_vals for w in ee_synw_vals]
    print(f"Total runs: {len(tasks)}")
    t0 = time.time()

    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(evaluate_pair)(p, w, base_pars) for p, w in tasks
    )

    df = pd.DataFrame(results)
    df.to_csv(save_csv, index=False)
    print(f"\nGrid complete. Saved to {save_csv}")
    print(f"Elapsed: {(time.time()-t0)/3600:.2f} hours")
    return df

# ===============================================================
# Run main
# ===============================================================
if __name__ == "__main__":
    df = run_2D_grid(ee_connectivity_vals, ee_synw_vals, base_pars, n_jobs=5)
    print(df.head())

