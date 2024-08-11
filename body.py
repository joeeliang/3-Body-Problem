import numpy as np
import pygame
import random

class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        r = random.randint(100, 200)
        g = random.randint(100, 200)
        b = random.randint(100, 200)
        self.colour = (r,g,b)

    def compute_acceleration(self, other_bodies, G=1.0, softening=0.2):
        acceleration = np.zeros(2)
        for other in other_bodies:
            if other is not self:
                r = other.position - self.position
                distance = np.linalg.norm(r)
                acceleration += G * other.mass * r / np.maximum(distance**3, softening**3)
        return acceleration
    
    def draw(self, frame, system, positions, win):
        scale = 200
        x = positions[frame, system.bodies.index(self), 0] * scale + 500
        y = 1000 - (positions[frame, system.bodies.index(self), 1] * scale + 500)
        pygame.draw.circle(win, self.colour, (x, y), 10)
        if frame > 2:
            updatedPoints = []
            for point in positions[:frame, system.bodies.index(self)]:
                x = point[0] * scale + 500
                y = 1000 - (point[1] * scale + 500)
                updatedPoints.append((x, y))
            pygame.draw.lines(win, self.colour, False, updatedPoints, 3)
    
    def draw_loop(self, system, positions, win):
        scale = 200
        updatedPoints = []
        for point in positions[:, system.bodies.index(self)]:
            x = point[0] * scale + 500
            y = 1000 - (point[1] * scale + 500)
            updatedPoints.append((x, y))
        pygame.draw.lines(win, self.colour, False, updatedPoints, 3)

    def get_state(self):
        return np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]])