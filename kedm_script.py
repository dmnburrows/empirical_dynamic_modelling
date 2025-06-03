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
import h5py
warnings.filterwarnings("ignore", category=RuntimeWarning) 
os.environ["OMP_DISPLAY_ENV"] = "FALSE"


#Import your modules
#---------------------------------------
import admin_functions as adfn
import CCM as cfn
# Define paths
#----------------------------------------------------------------------
code = '~/Documents/empirical_dynamic_modelling'
data = '/snl/scratch25/dburrows/CCM'

h5_li = np.sort(glob.glob(f'{data}/*trace*pre-CCM*h5'))
e_li = np.sort(glob.glob(f'{data}/*trace*pre-CCM*_E.npy'))

assert sum([os.path.basename(e_li[i]).split('CCM')[0] == os.path.basename(h5_li[i]).split('CCM')[0] 
for i in range(len(h5_li))]) == len(h5_li), 'mismatch in CCM and E files'

# Add it to the HDF5 input file
for i in range(len(h5_li)):
    print('Running ' + os.path.basename(h5_li[i]))
    if os.path.basename(e_li[i]).split('CCM')[0] == os.path.basename(h5_li[i]).split('CCM')[0]:
        with h5py.File(h5_li[i], 'r') as f:
            print(list(f.keys()))
            d=f['data'][:]

        e = np.load(e_li[i])

        result = kedm.xmap(dataset=d, edims=list(e.astype(int)), tau=int(1), Tp=int(0))  
        
        
        basename = os.path.basename(h5_li[i]).replace('pre-CCM.h5', 'CCMxmap.npy')
        save_path = os.path.join(os.path.dirname(h5_li[i]), basename)

        # Save result (adjust depending on what kedm.xmap returns)
        np.save(save_path, result)

        