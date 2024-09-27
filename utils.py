import numpy as np
import time
from functools import wraps
import pandas as pd

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds to execute.")
        return result
    return wrapper


def make_state(d, vx, vy, bodies=3):
    middle_state = [0,0,vx,vy]
    left = [d, 0, -middle_state[2]/2,-middle_state[3]/2]
    right = [-d, 0, -middle_state[2]/2,-middle_state[3]/2]
    return np.array(middle_state + left + right)

def rearrange(file_path):
    # Sample CSV data
    df = pd.read_csv(file_path)

    # Sort by vy and then by vx
    df_sorted = df.sort_values(by=['vy', 'vx'])
    # df_sorted[['vx', 'vy']] = df_sorted[['vy', 'vx']]

    df_sorted.to_csv(file_path, index=False)