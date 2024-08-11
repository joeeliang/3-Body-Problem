import csv
import numpy as np
import visualization

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
    with open('cut_' + filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for i in range(stop_row):
            for j in range(3):
                writer.writerow(rows[i * 3 + j])
            writer.writerow([])

def loop_csv():
    positions = visualization.read_csv('data/positions.csv', 3)
    min_step = proximity(positions)
    cut_csv('positions.csv', min_step + 1)
    visualization.pygame_animate("data/cut_positions.csv", loop = True)
    
