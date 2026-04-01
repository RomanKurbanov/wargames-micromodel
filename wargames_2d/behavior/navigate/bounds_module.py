import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class BoundsModule(SteeringModule):
    weight = 1.0

    def __init__(self, bounds_radius: float = 2000.0):
        self.bounds_radius = bounds_radius

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        dist = ship.position.length()
        inner = self.bounds_radius * 0.8
        if dist <= inner:
            return pygame.math.Vector2(0, 0)

        toward_center = safe_normalize(-ship.position)
        t = min((dist - inner) / (self.bounds_radius - inner + 1.0), 1.0)
        # Return a force up to magnitude 2 so weight=1 easily overpowers other modules
        return toward_center * (t * 2.0)
