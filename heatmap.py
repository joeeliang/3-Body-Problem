import numpy as np
import matplotlib.pyplot as plt
from visualization import pygame_run
from utils import make_state
import pandas as pd
import os

def plot_proximity_heatmap(vx_values, vy_values, proximity_values):
    # Reshape the proximity_values into a 2D array for heatmap plotting
    num_vx = len(np.unique(vx_values))
    num_vy = len(np.unique(vy_values))
    proximity_heatmap = proximity_values.reshape(num_vy, num_vx)
    threshold = 3.5
    # Apply binary threshold
    proximity_heatmap[proximity_heatmap >= threshold] = np.nan  # Set values above threshold to NaN
    
    vx_range = np.unique(vx_values)
    vy_range = np.unique(vy_values)
    
    extent = [vx_range.min(), vx_range.max(), vy_range.min(), vy_range.max()]
    cmap = plt.cm.get_cmap('hot')
    cmap.set_under('cyan') 
    plt.imshow(proximity_heatmap, origin='lower', cmap=cmap, extent=extent, aspect='auto', vmin=0.01)
    plt.colorbar(label='proximity')
    plt.xlabel('vx')
    plt.ylabel('vy')

    # # Function to format the tooltip text
    # def format_coord(x, y):
    #     if x >= vx_range.min() and x <= vx_range.max() and y >= vy_range.min() and y <= vy_range.max():
    #         x_index = int((x - vx_range.min()) / (vx_range.max() - vx_range.min()) * (num_vx))
    #         y_index = int((y - vy_range.min()) / (vy_range.max() - vy_range.min()) * (num_vy))
    #         return f'vx={vx_range[x_index]:.5f}, vy={vy_range[y_index]:.5f}, proximity={proximity_heatmap[y_index, x_index]:.6f}'
    #     else:
    #         return ''

    # plt.gca().format_coord = format_coord

    # Event handler for key press
    vx_selected = None
    vy_selected = None
    def on_key(event):
        nonlocal vx_selected, vy_selected
        if event.key == 'u':
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                if vx_range.min() <= x <= vx_range.max() and vy_range.min() <= y <= vy_range.max():
                    x_index = int((x- vx_range.min()) / (vx_range.max() - vx_range.min()) * (num_vx))
                    y_index = int((y- vy_range.min()) / (vy_range.max() - vy_range.min()) * (num_vy))
                    vx_selected = vx_range[x_index]
                    vy_selected = vy_range[y_index]
                    print(f'Pressed u at (vx, vy): {vx_selected}, {vy_selected}, {proximity_heatmap[y_index, x_index]}')
                    # Debug print before calling the animation
                    print("Calling animation.animate with:", vx_selected, vy_selected)
                    pygame_run(make_state(1, vx_selected, vy_selected,))
        if event.key == 'm':
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(x, y)
            


    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    plt.show()

data = pd.read_csv('/Users/joeliang/Joe/Coding/3-Body-Problem/multiprocessing/build/enhanced/swapped.csv')
vx_values = data['vx'].values
vy_values = data['vy'].values
proximity_values = data['proximity'].values
plot_proximity_heatmap(vx_values, vy_values, proximity_values)