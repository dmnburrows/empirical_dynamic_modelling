import time
import os
import subprocess
import kedm

star = time.time()
print('running kedm')

env = os.environ.copy()
env['HDF5_DISABLE_VERSION_CHECK'] = '1'

# Run the command
cmd = [
    "edm-xmap",
    "--dataset", "data",
    "--rho",
    "--rho-diff",
    "/snl/scratch25/dburrows/CCM/prc.h5",
    "/snl/scratch25/dburrows/CCM/prac_xmap.h5"
]

# Run it and wait for it to finish
subprocess.run(cmd, env=env, check=True)

end = time.time()
print(f'Time kedm = {end-star}')




star = time.time()
print('running pykedm')

import h5py

# Add it to the HDF5 input file
with h5py.File('/snl/scratch25/dburrows/CCM/prc.h5', 'a') as f:
    print(list(f.keys()))
    d=f['data'][:]
    em=f['e'][:]

kedm.xmap(d, edims=em, tau=1, Tp=0)  
  
end = time.time()
print(f'Time pykedm = {end-star}')
