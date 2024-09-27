import numpy as np
import time
from functools import wraps
import pandas as pd
import subprocess

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


def run(input2, dimensions):
    # Define the inputs
    input1 = str("normal" + "\n")
    input2 = str(input2 + "\n")
    input3 = str(dimensions)

    # Define the command to run the executable
    cmd = ["./app"]

    # Run the executable
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write the inputs to the process
    process.stdin.write(input1.encode())
    process.stdin.write(input2.encode())
    process.stdin.write(input3.encode())

    # Wait for the process to finish
    output, error = process.communicate()

    # Check if the program ran successfully
    if process.returncode == 0:
        print("Program ran successfully")
    else:
        print("Error running program:", error.decode())

    # You can also capture the output of the program if it prints anything to stdout
    print("Output:", output.decode())
    rearrange("data/zoom.csv")

def get_positions(state):
    input1 = str("positions" + "\n")
    input2 = str(state)

    # Define the command to run the executable
    cmd = ["./app"]

    # Run the executable
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write the inputs to the process
    process.stdin.write(input1.encode())
    process.stdin.write(input2.encode())

    # Wait for the process to finish
    output, error = process.communicate()

    # Check if the program ran successfully
    if process.returncode == 0:
        print("Program ran successfully")
    else:
        print("Error running program:", error.decode())

    # You can also capture the output of the program if it prints anything to stdout
    print("Output:", output.decode())
