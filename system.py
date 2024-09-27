import numpy as np
from utils import timer
import pygame

black = (0,0,0)  # RGB value for black
preset_colours = [(219, 255, 254),(255, 235, 205),(255, 80, 0)]

class Body:
    def __init__(self, mass, position, velocity, body_id):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.colour = preset_colours[body_id]

        # Pre-create the glow surface (done once during initialization)
        glow_size = 200
        self.glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
    
        glow_x = glow_size / 2
        glow_y = glow_x

        # Draw glow once
        for radius, alpha in zip(range(67, 0, -5), range(1, 30, 5)):
            pygame.draw.circle(self.glow_surface, (*self.colour, alpha), (glow_x, glow_y), radius)

    def draw(self, frame, system, positions, win):
        scale = 200
        x = positions[frame, system.bodies.index(self), 0] * scale + 500
        y = 1000 - (positions[frame, system.bodies.index(self), 1] * scale + 500)
        
        # Blit the pre-created glow surface onto the main window
        win.blit(self.glow_surface, (x - self.glow_surface.get_width() // 2, y - self.glow_surface.get_height() // 2))

        # Draw the celestial body (star)
        pygame.draw.circle(win, self.colour, (x, y), 10)

        # Create a transparent surface for the trail (supports alpha)
        trail_surface = pygame.Surface((win.get_width(), win.get_height()), pygame.SRCALPHA)

        # Draw the fading trail if frame > 2

        trail_length = positions.shape[0] - 1  # Limit the trail length to the last 50 points
        for i in range(trail_length, 1, -1):
            # Get two consecutive points to draw a segment
            point1 = positions[frame - i, system.bodies.index(self)]
            point2 = positions[frame - i + 1, system.bodies.index(self)]

            # Scale positions
            x1 = point1[0] * scale + 500
            y1 = 1000 - (point1[1] * scale + 500)
            x2 = point2[0] * scale + 500
            y2 = 1000 - (point2[1] * scale + 500)

            # Calculate fading alpha based on the age of the point
            fade_factor = int(255 * (1 - i / trail_length))
            
            # Draw the fading segment on the trail surface
            colour_with_alpha = (*self.colour[:3], fade_factor)  # Apply fade_factor to the alpha component
            pygame.draw.line(trail_surface, colour_with_alpha, (x1, y1), (x2, y2), 3)

        # Blit the trail surface onto the main window
        win.blit(trail_surface, (0, 0))

    def get_state(self):
        return np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]])

class System:
    def __init__(self, G=1.0, state=None, bodies=None):
        self.G = G
        if state is not None:
            self.bodies = []
            for body in range(int(len(state)/4)):
                n = int(body*4)
                self.bodies.append(Body(mass=1.0, position=[state[0+n], state[1+n]], velocity=[state[2+n], state[3+n]], body_id=body))
        else:
            self.bodies = bodies

    def compute_accelerations(self):
        accelerations = []
        for body in self.bodies:
            other_bodies = [b for b in self.bodies if b is not body]
            accelerations.append(body.compute_acceleration(other_bodies, self.G))
        return accelerations

    def compute_total_energy(self):
        kinetic_energy = 0.5 * sum(body.mass * np.dot(body.velocity, body.velocity) for body in self.bodies)
        potential_energy = 0
        for i, body1 in enumerate(self.bodies):
            for body2 in self.bodies[i+1:]:
                distance = np.linalg.norm(body2.position - body1.position)
                potential_energy -= self.G * body1.mass * body2.mass / distance
        return kinetic_energy + potential_energy

    def get_state(self):
        return np.concatenate([body.get_state() for body in self.bodies])

    def integrate(self, dt, num_steps, save_positions=False):
        # Save positions only if needed, don't waste storage.
        if save_positions:
            positions = np.zeros((num_steps, len(self.bodies), 2))
            delta_energy = np.zeros(num_steps)
            initial_energy = self.compute_total_energy()

        for step in range(num_steps):
            accelerations = self.compute_accelerations()
            
            for i, body in enumerate(self.bodies):
                # Update positions (Verlet scheme step 1)
                body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

            new_accelerations = self.compute_accelerations()
            
            for i, body in enumerate(self.bodies):
                # Update velocities (Verlet scheme step 3)
                body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt
            # Save positions only if needed, don't waste storage.
            if save_positions:
                positions[step] = [body.position for body in self.bodies]
                delta_energy[step] = initial_energy - self.compute_total_energy()
        if save_positions: 
            return positions, delta_energy
        else:
            # Normally just return the end
            return self.get_state()