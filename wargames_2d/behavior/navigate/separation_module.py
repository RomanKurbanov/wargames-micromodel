import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class SeparationModule(SteeringModule):
    weight = 0.5

    def __init__(self, neighbor_radius: float = 80.0):
        self.neighbor_radius = neighbor_radius

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        r      = self.neighbor_radius
        px, py = ship.position.x, ship.position.y
        r2     = r * r
        inv_ss = 1.0 / (ship.size if ship.size > 0.01 else 0.01)
        sx = sy = 0.0
        for other in registry.ships_grid.query(px, py, r):
            if other is ship:
                continue
            dx = other.position.x - px
            dy = other.position.y - py
            d2 = dx * dx + dy * dy
            if d2 > r2 or d2 < 1e-12:
                continue
            dist        = d2 ** 0.5
            size_factor = other.size * inv_ss
            strength    = (1.0 - dist / r) * size_factor
            inv_d       = 1.0 / dist
            sx -= dx * inv_d * strength
            sy -= dy * inv_d * strength
        steering = pygame.math.Vector2(sx, sy)
        if steering.length() > 1.0:
            steering = safe_normalize(steering)
        return steering
