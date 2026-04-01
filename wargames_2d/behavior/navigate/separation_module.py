import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class SeparationModule(SteeringModule):
    weight = 0.5

    def __init__(self, neighbor_radius: float = 80.0):
        self.neighbor_radius = neighbor_radius

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        steering = pygame.math.Vector2(0, 0)
        for other in registry.ships:
            if other is ship:
                continue
            to_other = other.position - ship.position
            dist = to_other.length()
            if dist > self.neighbor_radius or dist < 1e-6:
                continue
            size_factor = other.size / max(ship.size, 0.01)
            strength = (1.0 - dist / self.neighbor_radius) * size_factor
            steering -= safe_normalize(to_other) * strength
        if steering.length() > 1.0:
            steering = safe_normalize(steering)
        return steering
