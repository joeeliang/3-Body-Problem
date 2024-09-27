import pygame
import matplotlib.pyplot as plt
from system import System
from matplotlib.animation import FuncAnimation
import numpy as np
import csv

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