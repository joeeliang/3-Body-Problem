from chaos import lyapunov, proximity
from visualization import pygame_run, plot_energy, animate_matlab
from utils import make_state
import numpy as np
import pandas as pd
from tqdm import tqdm

def heatmap(quantifier):    
    # Define the ranges for d, vx, and vy
    vx_values = np.linspace(0, 1, 60)
    vy_values = np.linspace(0, 1, 60)

    # Create an empty list to store the results
    results_list = []

    # Total number of iterations
    total_iterations = len(vx_values) * len(vy_values)

    # Iterate over the grid points with a progress bar
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        for vx in vx_values:
            for vy in vy_values:
                initial_state = make_state(1, vx, vy)
                lyapunov_exponent = quantifier(initial_state, total_time=20)
                results_list.append({"d": 1, "vx": vx, "vy": vy, "lyapunov_exponent": " " + str(lyapunov_exponent)})
                pbar.update(1)
        # Convert the results list to a DataFrame
    results = pd.DataFrame(results_list)

    # Save the results to a CSV file
    results.to_csv("lyapunov_exponents_3d.csv", index=False)

if __name__ == "__main__":
    # Run your desired simulations or visualizations here
    pygame_run(make_state(1,0.2033898305084746,0.05084745762711865))
    heatmap(proximity)
