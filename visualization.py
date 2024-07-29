import pygame
import matplotlib.pyplot as plt
from system import System
from matplotlib.animation import FuncAnimation

def pygame_run(initial):
    system=System(state=initial)
    # Integrate system
    num_steps = 2000
    dt = 0.01
    positions, delta_energy =  system.integrate(dt, num_steps,save_positions=True)
    class Canvas:
        def __init__(self):
            self.canvasX = 0
            self.canvasY = 0

        def move(self, dx, dy):
            self.canvasX += dx
            self.canvasY += dy

    canvas = Canvas()
    WIN = pygame.display.set_mode((1000, 1000))

    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_i:
                    waiting = False

    for i in range(positions.shape[0]):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    pygame.quit()

        # Draw everything
        WIN.fill((10,10,10))
        for body in system.bodies:
            body.draw(i, system, positions, WIN)

        # Update the display
        pygame.display.flip()
        pygame.time.delay(20)

def plot_energy(delta_energy):
    plt.plot(range(len(delta_energy)), delta_energy)
    plt.xlabel('Step')
    plt.ylabel('Energy Difference')
    plt.title('Energy Conservation Over Time')
    plt.show()

def animate_matlab(positions):
    fig, ax = plt.subplots()
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('3-Body Problem Trajectories')

    line1, = ax.plot([], [] , label='Body 1')
    line2, = ax.plot([], [], label='Body 2')
    line3, = ax.plot([], [], label='Body 3')
    ax.legend()

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def update(frame):
        line1.set_data(positions[:frame, 0, 0], positions[:frame, 0, 1])
        line2.set_data(positions[:frame, 1, 0], positions[:frame, 1, 1])
        line3.set_data(positions[:frame, 2, 0], positions[:frame, 2, 1])
        return line1, line2, line3

    ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, interval=10)

    plt.show()