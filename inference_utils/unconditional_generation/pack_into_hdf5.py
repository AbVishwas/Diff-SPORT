import numpy as np
import h5py
import sys
import os
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # Unlock the h5py file

# Usage: python pack_into_hdf5.py <config_name> <npy_file1> <npy_file2> ...
config_name = sys.argv[1]
inp_files = sys.argv[2:]

assert any(config_name in string for string in inp_files), "Mismatch in config and input names, please check again!!"

# Resolve repository root and default output directory
REPO_ROOT = Path(__file__).resolve().parents[2]
default_out = REPO_ROOT / 'inference_utils' / 'unconditional_generation' / 'generated_samples'
default_out.mkdir(parents=True, exist_ok=True)

out_path = default_out / f"generated_samples-25k-{config_name}_ep1000.h5"
with h5py.File(str(out_path), "w") as h5data:
    for idx, inp_file in enumerate(inp_files):
        arr_path = Path(inp_file)
        if not arr_path.is_absolute():
            arr_path = REPO_ROOT / arr_path
        arr_ = np.load(str(arr_path))
        print(f"loaded file {idx+1}, shape={arr_.shape}")
        h5data.create_dataset('seed_'+str(idx+1), data=arr_)
        print(f"finished file {idx+1}")

print(f"Packed {len(inp_files)} arrays into {out_path}")
