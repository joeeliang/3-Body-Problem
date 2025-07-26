import numpy as np
import time
from functools import wraps
import pandas as pd
import subprocess
import csv
import visualization

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

def calculate_positions(state, own_state= False):
    if own_state:
        input1 = str("state" + "\n")
    else:
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


def proximity(positions):
    initial_state = positions[0].flatten()
    min_distance = float("inf")
    min_step = 0
    di = 0
    for i, frame in enumerate(positions):
        state = frame.flatten()
        distance = np.linalg.norm(state - initial_state)
        if distance < min_distance and distance < di:
            min_distance = distance
            min_step = i
        di = distance
    return min_step

def cut_csv(filename, stop_row):
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        rows = []
        row_num = 0
        for row in reader:
            if row:  # check if the row is not empty
                rows.append(row)
                row_num += 1
            else:
                row_num += 1
            if row_num > stop_row * 3 + stop_row - 1:
                break
    print(rows)
    with open('data/cut_positions.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(stop_row):
            for j in range(3):
                writer.writerow(rows[i * 3 + j])
            writer.writerow([])

def loop_csv():
    '''Creates new CSV file with the positions so that it is looped as closest as it can be. Truncated at point of closest proximinty.'''
    positions = visualization.read_csv('data/positions.csv', 3)
    min_step = proximity(positions)
    cut_csv('data/positions.csv', min_step + 1)
    
