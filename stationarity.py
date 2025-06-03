import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from pyEDM import EmbedDimension, Simplex
from multiprocessing import Pool, cpu_count
from pprint import pprint
import os
#Import packages
#---------------------------------------
import sys
import os
import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import matplotlib
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

#Import your modules
#---------------------------------------
import admin_functions as adfn
import CCM as cfn
# Define paths
#----------------------------------------------------------------------
code = '~/Documents/empirical_dynamic_modelling'
data = '/snl/scratch25/dburrows/CCM'

sys.version

all_results = []

prac = glob.glob(f'{data}/*regtrace.npy')
for p in prac:
    print(f"Processing file: {p}")
    curr = np.load(p)
    curr_T = curr.T
    scaler = StandardScaler()
    curr_T_scaled = scaler.fit_transform(curr_T)
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(curr_T_scaled)
    
    
    # Step 1: Embed dimension estimation in parallel
    def compute_embed_dim(i):
        try:
            df = pd.DataFrame({
                'Time': np.arange(pcs.shape[0]),
                'X': pcs[:, i]
            })

            result = EmbedDimension(
                dataFrame=df,
                columns='X',
                target='X',
                lib=f'1 {pcs.shape[0]//2}',       # first half
                pred=f'{pcs.shape[0]//2} {pcs.shape[0]}',   # second half
                maxE=20,
                tau=-1,
                Tp=1,
                showPlot=False,
                numThreads=1
            )
            best_E = result['E'][np.argmax(result['rho'])]
            return best_E
        except Exception as e:
            return f"Error for PC {i}: {e}"

    with Pool(min(10, cpu_count())) as pool:
        embed_dims = pool.map(compute_embed_dim, list(range(10)))

    # Step 2: Simplex prediction on each PC
    for i in range(10):
        E = embed_dims[i]
        pc_series = pcs[:, i]
        T = len(pc_series)
        third = T // 3

        df = pd.DataFrame({
            'Time': np.arange(T),
            'X': pc_series
        })

        for target_chunk, name in [
            (range(0, third), 'self'),
            (range(third, 2 * third), '2nd'),
            (range(2 * third, T), '3rd')
        ]:
            pred_range = f"{target_chunk.start + 1} {target_chunk.stop}"
            lib_range = f"1 {third}"

            simplex_result = Simplex(
                dataFrame=df,
                columns='X',
                target='X',
                lib=lib_range,
                pred=pred_range,
                E=int(E),
                tau=-1,
                Tp=1
            )

            rho = simplex_result.dropna(subset=["Observations", "Predictions"])["Observations"].corr(simplex_result["Predictions"])

            all_results.append({
                'File': os.path.basename(p),
                'PC': i + 1,
                'E': int(E),
                'TargetChunk': name,
                'rho': rho
            })

# Final DataFrame and save
df_all = pd.DataFrame(all_results)
df_all.to_csv(f'{data}/all_stationarity_results_scaled.csv', index=False)
print("\n✅ Saved all results to 'all_simplex_results.csv'")
