import numpy as np
import matplotlib.pyplot as plt
import visualization 
import utils
import pandas as pd
from closest_position import loop_csv
import os

def plot_proximity_heatmap(data_path,next=False):
    data = pd.read_csv(data_path)
    vx_values = data['vx'].values
    vy_values = data['vy'].values
    proximity_values = data['proximity'].values

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
    plt.imshow(proximity_heatmap, origin='lower', cmap=cmap, extent=extent, aspect='auto', vmin=0.002)
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
                    str_arr = ' '.join(map(str, [1, vx_selected,vy_selected]))
                    utils.get_positions(str_arr)
                    visualization.pygame_animate("data/positions.csv")
        if event.key == 'm':
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(x, y)
        if event.key == 'h':
            plt.axis([0,1,0,1])
        if event.key == 'n':
            print("Doing loop")
            loop_csv()
            visualization.pygame_animate("data/cut_positions.csv")
        if event.key == "i":
            xLim = plt.gca().get_xlim()
            yLim = plt.gca().get_ylim()
            plt.close()
            print("Generating new plot")
            enhance(xLim, yLim)
            print("done")
        if event.key == 'z':
            xLim = plt.gca().get_xlim()
            yLim = plt.gca().get_ylim()
            x, y = find_minimum_proximity("data/zoom.csv", xLim, yLim)
            zoom_factor = abs(xLim[0]-xLim[1])/4
            zoom(x, y, zoom_factor)
            plt.draw()
        if event.key == "a":
            auto(ax)

    ax = plt.gca()
    plt.gcf().canvas.mpl_connect('key_press_event', on_key)
    if next:
        plt.show(block=False)
    else:
        plt.show()
    return ax

def zoom(x, y ,factor):
    plt.axis([x - factor, x + factor, y - factor, y + factor])

def auto(ax):
    for a in range(6):
        print(a)
        plt.sca(ax)
        xLim = plt.gca().get_xlim()
        yLim = plt.gca().get_ylim()
        x, y = find_minimum_proximity("data/zoom.csv", xLim, yLim)
        zoom_factor = abs(xLim[0]-xLim[1])/4
        zoom(x, y, zoom_factor)
        plt.draw()
        xLim = plt.gca().get_xlim()
        yLim = plt.gca().get_ylim()
        plt.close()
        print("Generating new plot")
        ax = enhance(xLim, yLim)
        plt.sca(ax)
        print("done")
    plt.close()
    plot_proximity_heatmap("data/zoom.csv")

def find_minimum_proximity(csv_file, xLim, yLim):
    df = pd.read_csv(csv_file)
    df_filtered = df[(df['vx'] >= xLim[0]) & (df['vx'] <= xLim[1]) & (df['vy'] >= yLim[0]) & (df['vy'] <= yLim[1])]
    min_proximity_row = df_filtered.loc[df_filtered['proximity'].idxmin()]
    print(df_filtered['proximity'].min())
    x, y = min_proximity_row['vx'], min_proximity_row['vy']
    print(x, y)
    return x, y

def enhance(xLim, yLim):
    input1 = f"{xLim[0]} {xLim[1]} {yLim[0]} {yLim[1]}"
    utils.run(input1, 50)
    ax = plot_proximity_heatmap("data/zoom.csv", next=True)
    print("hello")
    return ax

def get_axis_limits(ax):
    return ax.get_xlim(), ax.get_ylim()

if __name__ == '__main__':
    plot_proximity_heatmap('data/zoom.csv')