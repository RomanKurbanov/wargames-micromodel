import pygame
from behavior.steering_module import SteeringModule


class EnemyBaseModule(SteeringModule):
    def __init__(self, detection_range: float = 600.0):
        self.detection_range = detection_range

    def get_enemies(self, ship, registry) -> list:
        faction = ship.faction
        r       = self.detection_range
        px, py  = ship.position.x, ship.position.y
        r2      = r * r
        enemies = []
        for other in registry.ships_grid.query(px, py, r):
            if other is ship or other.faction == faction:
                continue
            dx = other.position.x - px
            dy = other.position.y - py
            if dx * dx + dy * dy <= r2:
                enemies.append(other)
        return enemies

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        return pygame.math.Vector2(0, 0)
