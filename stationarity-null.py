import glob
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
from pyEDM import EmbedDimension, Simplex
from multiprocessing import Pool, cpu_count
import os
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import custom modules
import admin_functions as adfn
import CCM as cfn

# Define paths
code = '~/Documents/empirical_dynamic_modelling'
data = '/snl/scratch25/dburrows/CCM'

all_results = []

# Get list of all trace files
prac = glob.glob(f'{data}/*regtrace.npy')
for p in prac:
    print(f"Processing file: {p}")
    
    # Load and preprocess data
    curr = np.load(p)
    curr_T = curr.T
    scaler = StandardScaler()
    curr_T_scaled = scaler.fit_transform(curr_T)
    
    # Run PCA
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
                lib=f'1 {pcs.shape[0]//2}',
                pred=f'{pcs.shape[0]//2} {pcs.shape[0]}',
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

    # Step 2: Simplex prediction using random thirds
    for i in range(10):
        E = embed_dims[i]
        if isinstance(E, str):  # Skip if there was an error
            print(E)
            continue

        pc_series = pcs[:, i]
        T = len(pc_series)

        df = pd.DataFrame({
            'Time': np.arange(T),
            'X': pc_series
        })

        # Select two random, non-overlapping thirds
        third = T // 3
        indices = np.arange(T)
        np.random.seed(42 + i)  # reproducible randomness per PC
        np.random.shuffle(indices)

        lib_idx = np.sort(indices[:third])
        pred_idx = np.sort(indices[third:2*third])

        lib_range = f"{lib_idx[0]+1} {lib_idx[-1]+1}"
        pred_range = f"{pred_idx[0]+1} {pred_idx[-1]+1}"

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

        # Calculate rho
        valid = simplex_result.dropna(subset=["Observations", "Predictions"])
        rho = valid["Observations"].corr(valid["Predictions"])

        all_results.append({
            'File': os.path.basename(p),
            'PC': i + 1,
            'E': int(E),
            'TargetChunk': 'random_third_vs_third',
            'rho': rho
        })

# Save final results
df_all = pd.DataFrame(all_results)
df_all.to_csv(f'{data}/all_stationarity_results_scaled_null.csv', index=False)
print("\n✅ Saved all results to 'all_stationarity_results_scaled.csv'")
