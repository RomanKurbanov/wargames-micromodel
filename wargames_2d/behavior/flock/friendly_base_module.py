import pygame
from behavior.steering_module import SteeringModule


class FriendlyBaseModule(SteeringModule):
    def __init__(self, neighbor_radius: float = 200.0):
        self.neighbor_radius = neighbor_radius

    def get_neighbors(self, ship, registry) -> list:
        r2     = self.neighbor_radius ** 2
        px, py = ship.position.x, ship.position.y
        neighbors = []
        for other in registry.ships_by_faction.get(ship.faction, []):
            if other is ship:
                continue
            dx = other.position.x - px
            dy = other.position.y - py
            if dx * dx + dy * dy <= r2:
                neighbors.append(other)
        return neighbors

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        return pygame.math.Vector2(0, 0)
