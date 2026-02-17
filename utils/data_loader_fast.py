# data_loader.py (drop-in replacement)
import os
import numpy as np

def get_dir(name):
    if name in ('ASH', 'RSS'):
        return 'data'
    return 'data'

def _load_array_fast(path, skiprows=3, cache=True, mmap=True):
    npy_path = path + '.npy'

    # 1) Binary cache for instant reloads
    if cache and os.path.exists(npy_path) and os.path.getmtime(npy_path) >= os.path.getmtime(path):
        return np.load(npy_path, mmap_mode='r' if mmap else None)

    arr = None

    # 2) Try pandas (usually 5–20x faster than loadtxt)
    try:
        import pandas as pd
        # whitespace-separated, no header, skip first 3 rows
        df = pd.read_csv(
            path,
            skiprows=skiprows,
            header=None,
            sep=r"\s+",
            engine="c",
            dtype=np.float64
        )
        arr = df.to_numpy(dtype=np.float64, copy=False)
    except Exception:
        pass

    # 3) Pure-NumPy fast path using fromstring (very fast, but reads into memory once)
    if arr is None:
        try:
            with open(path, "r") as f:
                # skip header lines
                for _ in range(skiprows):
                    next(f)
                first = f.readline()
                ncols = len(first.split())
                # concatenate first data line with the rest
                s = first + f.read()
            flat = np.fromstring(s, sep=" ", dtype=np.float64)
            rows = flat.size // ncols
            arr = flat[:rows * ncols].reshape(rows, ncols)
        except Exception:
            arr = None

    # 4) Last resort: np.loadtxt (slowest)
    if arr is None:
        arr = np.loadtxt(path, skiprows=skiprows, dtype=np.float64)

    # Save binary cache for future runs
    if cache:
        try:
            # Save as contiguous float64 so next time np.load(..., mmap_mode='r') is instant
            np.save(npy_path, np.ascontiguousarray(arr, dtype=np.float64))
        except Exception:
            pass

    return arr


def data_loader(dataset_name):
    # choose folder
    if 'RSS' in dataset_name:
        dataset_file_path = os.path.join(get_dir('RSS'), dataset_name)
    elif 'ASH' in dataset_name:
        dataset_file_path = os.path.join(get_dir('ASH'), dataset_name)
    else:
        dataset_file_path = os.path.join(get_dir('RSS'), dataset_name)

    # FAST replacement for the slow np.loadtxt line
    data = _load_array_fast(dataset_file_path, skiprows=3, cache=True, mmap=True)

    dim = data.shape[1] - 1
    num_steps = data.shape[0]
    return env_laoded_from_CSV(data), num_steps, dim


class env_laoded_from_CSV():
    def __init__(self, data):
        # Works with regular ndarray or memmap from np.load(..., mmap_mode='r')
        self.X = data[:, 1:]
        self.Y = data[:, 0]
        self.iter = 0

    def next(self):
        x = self.X[self.iter, :]
        y = self.Y[self.iter]
        self.iter += 1
        return x, y

    def get_all_data_at_once(self):
        return self.X, self.Y
