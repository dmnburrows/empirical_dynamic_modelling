#Import packages
#---------------------------------------
import sys
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import kedm
import warnings
import gc
from multiprocessing import Pool
from tqdm import tqdm

#Import your modules
#---------------------------------------
import CCM as cfn
import proc_functions as pfn

sys.path.insert(0, os.path.expanduser('~/Documents/admin_tools'))
from admin_tools import admin_functions as adfn

# Define paths
#----------------------------------------------------------------------
code_path = '~/Documents/empirical_dynamic_modelling'
data_path = '/snl/scratch25/dburrows/CCM'
mask_l = np.sort(glob.glob(f'{data_path}/*mask*'))

sys.version


# confirm -> on sparse data - e.g. subsample
#=========================================
#participation ratio - use PCA


def compute_pr(data):

    dic = cfn.xtr(data)
    X = dic['trace'].T
    scale = StandardScaler().fit_transform(X)
    pca = PCA().fit(scale)
    eigvals = pca.explained_variance_
    d_pr = (eigvals.sum()**2) / np.sum(eigvals**2)

    if '_sz_' in data or '_pre_' in data:cond = data.split('_')[-2]
    else:cond = data.split('_')[-3]
    name = dic['name']
    
    return(name, cond, d_pr)


    
    
if __name__ == "__main__":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    n_proc = 8
    print(f"Running on {n_proc} cores...")

    with Pool(processes=n_proc) as pool:
        results = list(tqdm(pool.imap_unordered(compute_pr, mask_l), total=len(mask_l)))

    # unpack results into dataframe
    name_l, cond_l, data_l = zip(*results)
    fin = pd.DataFrame({'pr': data_l, 'name': name_l, 'cond': cond_l})
    fin.to_csv(f"{data_path}/participation_ratio_PCA.csv", index=False)
    print("Saved:", f"{data_path}/participation_ratio_PCA.csv")
    