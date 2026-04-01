import pygame
from behavior.steering_module import SteeringModule


class EnemyBaseModule(SteeringModule):
    def __init__(self, detection_range: float = 600.0):
        self.detection_range = detection_range

    def get_enemies(self, ship, registry) -> list:
        enemies = []
        for other in registry.ships:
            if other is ship or other.faction == ship.faction:
                continue
            if (other.position - ship.position).length() <= self.detection_range:
                enemies.append(other)
        return enemies

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        return pygame.math.Vector2(0, 0)
