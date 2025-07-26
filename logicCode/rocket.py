import pygame
import random
import numpy as np

preset_colours = [(219, 255, 254), (255, 235, 205), (255, 80, 0)]

# New Star class
class Star:
    def __init__(self, x, y, z, brightness):
        self.x = x
        self.y = y
        self.z = z  # depth for scaling
        self.brightness = brightness

    def draw(self, win, offset_x, offset_y, scale):
        # Calculate the center of the window
        center_x, center_y = win.get_width() // 2, win.get_height() // 2
        
        # Apply scaling based on the z-coordinate
        scale_factor = scale / (scale + self.z)
        
        # Calculate scaled offsets
        scaled_offset_x = offset_x * scale_factor
        scaled_offset_y = offset_y * scale_factor

        # Calculate screen position relative to the center
        x = center_x + (self.x - center_x) * scale_factor + scaled_offset_x
        y = center_y + (self.y - center_y) * scale_factor - scaled_offset_y

        # Wrap around the screen
        x = x % win.get_width()
        y = y % win.get_height()

        # Scale the star size and brightness
        size = max(1, int(2 * scale_factor))
        color = int(self.brightness * scale_factor)
        color = max(0, min(color, 255))  # Ensure color is within valid range

        pygame.draw.circle(win, (color, color, color), (int(x), int(y)), size)

class Body:
    def __init__(self, mass, position, velocity, is_rocket=False):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.colour = preset_colours[2] if not is_rocket else (0, 0, 200)
        self.is_rocket = is_rocket
        self.thrust = 0.0001 if is_rocket else 0
        self.angle = 0 if is_rocket else None
        self.sparks = []

        glow_size = 200
        self.glow_surface = pygame.Surface((glow_size, glow_size), pygame.SRCALPHA)
    
        glow_x = glow_size / 2
        glow_y = glow_x

        for radius, alpha in zip(range(67, 0, -5), range(1, 30, 5)):
            pygame.draw.circle(self.glow_surface, (*self.colour, alpha), (glow_x, glow_y), radius)
        
        self.position_history = []

    def draw(self, win, rocket_pos, scale):
        # Calculate the relative position of the current body to the rocket
        relative_position = self.position - rocket_pos

        # Scale and translate the positions so the rocket is always in the center
        x = relative_position[0] * scale + win.get_width() // 2
        y = win.get_height() // 2 - (relative_position[1] * scale)

        if self.is_rocket:
            if self.thrust > 0:
                self.draw_sparks(win, x +  20 * np.sin(np.radians(self.angle))*scale/10000, y +  20 * np.cos(np.radians(self.angle))*scale/10000, scale/10000)

            rocket_image = pygame.Surface((20, 40), pygame.SRCALPHA)
            pygame.draw.polygon(rocket_image, self.colour, [(10, 0), (0, 40), (20, 40)])
            rocket_image = pygame.transform.rotozoom(rocket_image, self.angle, scale/10000)  # Scale the rocket image
            win.blit(rocket_image, rocket_image.get_rect(center=(x, y)))

        else:
            win.blit(self.glow_surface, (x - self.glow_surface.get_width() // 2, y - self.glow_surface.get_height() // 2))
            pygame.draw.circle(win, self.colour, (x, y), scale/15) # draw planet

        # Draw trail
        if len(self.position_history) > 2 and not self.is_rocket:
            points = []
            for pos in self.position_history:
                rel_pos = pos - rocket_pos
                trail_x = rel_pos[0] * scale + win.get_width() // 2
                trail_y = win.get_height() // 2 - (rel_pos[1] * scale)
                points.append((trail_x, trail_y))
            pygame.draw.lines(win, self.colour, False, points, 2)

    def draw_sparks(self, win, x, y, scale):
        
        # Create new sparks
        for _ in range(5):  # Add 5 new sparks each frame
            spark_x = x 
            spark_y = y 
            spark_speed = random.uniform(1, 3) * scale
            spark_angle = self.angle + random.uniform(-30, 30)
            spark_life = random.randint(10, 20)
            self.sparks.append([(spark_x, spark_y), spark_speed, spark_angle, spark_life])

        # Update and draw existing sparks
        new_sparks = []
        for spark in self.sparks:
            pos, speed, angle, life = spark
            # Move the spark
            new_x = pos[0] + speed * np.sin(np.radians(angle))
            new_y = pos[1] + speed * np.cos(np.radians(angle))
            # Decrease life and add to new list if still alive
            life -= 1
            if life > 0:
                new_sparks.append([(new_x, new_y), speed, angle, life])
                # Draw the spark
                color = (255, 255, 0) if life > 10 else (255, 165, 0)  # Yellow fading to orange
                pygame.draw.circle(win, color, (int(new_x), int(new_y)), max(2 * scale, 1))

        self.sparks = new_sparks

    def compute_acceleration(self, other_bodies, G=1.0, softening=0.18):
        acceleration = np.zeros(2)
        if not self.is_rocket:
            for other in other_bodies:
                if other is not self:
                    r = other.position - self.position
                    distance = np.linalg.norm(r)
                    acceleration += G * other.mass * r / np.maximum(distance**3, softening**3)
        
        if self.is_rocket:
            thrust_acceleration = np.array([-self.thrust * np.sin(np.radians(self.angle)),
                                            self.thrust * np.cos(np.radians(self.angle))])
            acceleration += thrust_acceleration
        
        return acceleration

    def get_state(self):
        state = np.array([self.position[0], self.position[1], self.velocity[0], self.velocity[1]])
        return state

    def update_rocket_controls(self, keys):
        if self.is_rocket:
            if keys[pygame.K_LEFT]:
                self.angle += 5
            if keys[pygame.K_RIGHT]:
                self.angle -= 5
            if keys[pygame.K_UP]:
                self.thrust = 10
            else:
                self.thrust = 0
            if keys[pygame.K_DOWN]:
                self.velocity = 0

class System:
    def __init__(self, G=1.0, state=None, bodies=None):
        self.G = G
        if state is not None:
            self.bodies = []
            for body in range(int(len(state)/4)):
                n = int(body*4)
                self.bodies.append(Body(mass=1.0, position=[state[0+n], state[1+n]], velocity=[state[2+n], state[3+n]]))
        else:
            self.bodies = bodies

    def compute_accelerations(self):
        accelerations = []
        for body in self.bodies:
            other_bodies = [b for b in self.bodies if b is not body]
            accelerations.append(body.compute_acceleration(other_bodies, self.G))
        return accelerations

    def get_state(self):
        state = []
        for body in self.bodies:
            body_state = body.get_state()
            state.extend(body_state)
        return np.array(state)

    def integrate(self, dt):
        accelerations = self.compute_accelerations()
        
        for i, body in enumerate(self.bodies):
            body.position += body.velocity * dt + 0.5 * accelerations[i] * dt**2

        new_accelerations = self.compute_accelerations()
        
        for i, body in enumerate(self.bodies):
            body.velocity += 0.5 * (accelerations[i] + new_accelerations[i]) * dt
            body.position_history.append(body.position.copy())
            if len(body.position_history) > 200000:  # Limit trail length
                body.position_history.pop(0)

def generate():
    body1 = Body(mass=1.0, position=[0.0, 0.0], velocity=[0.687546, 1.06785])
    body2 = Body(mass=1.0, position=[-1.0, 0.0], velocity=[-0.343773, -0.533925])
    body3 = Body(mass=1.0, position=[1.0, 0], velocity=[-0.343773, -0.533925])
    body4 = Body(mass=0.0001, position=[0.0, 0.3], velocity=[0.0, 0.0], is_rocket=True)

    system = System(bodies=[body1, body2, body3, body4])
    return system

def pygame_run(system):
    pygame.init()
    dt = 0.001
    
    WIN = pygame.display.set_mode((1000, 1000))
    font = pygame.font.Font(None, 24)
    clock = pygame.time.Clock()
    running = True
    scale = 10000

    # Generate stars with z-coordinate
    stars = [Star(random.randint(0, WIN.get_width()), 
                  random.randint(0, WIN.get_height()),
                  random.randint(1, 1000),  # z-coordinate
                  random.randint(100, 255)) for _ in range(200)]
    
    # Initialize star offset
    star_offset_x = 0
    star_offset_y = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        system.bodies[3].update_rocket_controls(keys)

        system.integrate(dt)

        WIN.fill((10, 10, 10))
        
        # Update star positions based on rocket movement
        if keys[pygame.K_UP]:
            star_offset_x += 0.5 * np.sin(np.radians(system.bodies[3].angle))
            star_offset_y -= 0.5 * np.cos(np.radians(system.bodies[3].angle))

        # Draw stars with scale
        for star in stars:
            star.draw(WIN, star_offset_x, star_offset_y, scale)

        rocket_pos = system.bodies[3].position
        if keys[pygame.K_PLUS] or keys[pygame.K_EQUALS]:
            scale *= 1.1
        if keys[pygame.K_MINUS]:
            scale /= 1.1
        for body in system.bodies:
            body.draw(WIN, rocket_pos, scale)

        # Draw the coordinate label
        x, y = pygame.mouse.get_pos()
        text = font.render(f'X: {rocket_pos[0]:.5f}, Y: {rocket_pos[1]:.5f}', True, (255, 255, 255))
        WIN.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
# Run the simulation
pygame_run(generate())