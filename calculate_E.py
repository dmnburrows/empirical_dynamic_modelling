import numpy as np
import pandas as pd
from pyEDM import EmbedDimension
from multiprocessing import Pool, cpu_count
import glob
import h5py
import os

data='/snl/scratch25/dburrows/CCM'
data_l = glob.glob(f'{data}/*trace*pre*CCM*h5')
    
# --- Function to process one neuron ---
def run_embed(re):
    df = pd.DataFrame({
        'Time': np.arange(len(re)),
        'X': re
    })

    result = EmbedDimension(
        dataFrame=df,
        columns='X',
        target='X',
        lib=f'1 {df.shape[0]//2}',
        pred=f'{df.shape[0]//2} {df.shape[0]}',
        maxE=20,
        tau=-1,
        Tp=1,
        showPlot=False,
        numThreads=1  # avoid nested parallelism
    )
    best_E = result['E'][np.argmax(result['rho'])]
    return best_E


# --- Main parallel code ---
if __name__ == '__main__':


    for x in data_l:
        print('Running ' + x)
        curr = x
        # Add it to the HDF5 input file
        with h5py.File(curr, 'r') as f:
            d=f['data'][:]
            
        args = [d[:,s] for s in range(d.shape[1])]

        with Pool(processes=110) as pool:
            results = pool.map(run_embed, args)

        fname = os.path.basename(x).replace('.h5', '_E.npy')
        np.save(os.path.join(data, fname), results)
        print(f"Saved {fname}")