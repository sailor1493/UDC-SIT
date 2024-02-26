import os
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def compare(val_path, test_arrs):
    val_arr = np.load(val_path)
    for test_arr, test_path in test_arrs:
        diff = np.abs(test_arr - val_arr)
        max_diff = np.max(diff)
        if max_diff < 0.1:
            print(f"Max diff: {max_diff}, test: {test_path}, val: {val_path}")
            break


val_paths = []
for i in range(1, 10):
    val_set = f"/home/n2/chanwoo/UDC/cvpr_rebuttal/DISCNet/datasets/synthetic_data/input/ZTE_new_{i}/train"
    val_files = os.listdir(val_set)
    val_files.sort()
    val_paths.extend([os.path.join(val_set, k) for k in val_files])
# val_arrs = [(np.load(k), k) for k in val_paths]

test_set = "/home/n2/chanwoo/input"
test_files = os.listdir(test_set)
test_files.sort()
test_paths = [os.path.join(test_set, k) for k in test_files]
test_arrs = [(np.load(k), k) for k in test_paths]

pool = mp.Pool(mp.cpu_count())
pool.starmap(compare, [(val_path, test_arrs) for val_path in val_paths])
pool.close()
pool.join()
