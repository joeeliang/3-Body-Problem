import numpy as np
from body import Body
from utils import timer

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