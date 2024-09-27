import pygame
import matplotlib.pyplot as plt
from system import System
from matplotlib.animation import FuncAnimation
import numpy as np
import csv
import utils
import pandas as pd


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
                    pygame_animate("data/positions.csv")
        if event.key == 'm':
            x, y = event.xdata, event.ydata
            if x is not None and y is not None:
                print(x, y)
        if event.key == 'h':
            plt.axis([0,1,0,1])
        if event.key == 'n':
            print("Doing loop")
            utils.loop_csv()
            pygame_animate("data/cut_positions.csv")
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

def read_csv(filename, num_bodies):
    positions = []
    with open(filename, 'r') as file:
        reader = csv.reader(file)
        frames = []
        for row in reader:
            if row:  # check if the row is not empty
                frames.append([float(val) for val in row])
            else:
                if frames:  # check if frames is not empty
                    positions.append(frames)
                    frames = []
        if frames:  # check if frames is not empty
            positions.append(frames)
    positions = np.array(positions)
    
    return positions

def pygame_animate(positions_path):
    num_bodies = 3  # assuming 3 bodies in your system
    positions = read_csv(positions_path, num_bodies)

    system=System(state=np.zeros(12))
    class Canvas:
        def __init__(self):
            self.canvasX = 0
            self.canvasY = 0

        def move(self, dx, dy):
            self.canvasX += dx
            self.canvasY += dy

    canvas = Canvas()
    WIN = pygame.display.set_mode((1000, 1000))
    print(positions.shape)

    for i in range(positions.shape[0] * 10): # the integer after is how many times the thing is looped.
        WIN.fill((10,10,10))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    pygame.quit()
        for body in system.bodies:
            body.draw(i % positions.shape[0], system, positions, WIN)
        pygame.display.flip()
        pygame.time.delay(20)

def plot_energy(delta_energy):
    plt.plot(range(len(delta_energy)), delta_energy)
    plt.xlabel('Step')
    plt.ylabel('Energy Difference')
    plt.title('Energy Conservation Over Time')
    plt.show()