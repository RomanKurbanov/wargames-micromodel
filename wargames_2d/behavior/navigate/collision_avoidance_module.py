import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class CollisionAvoidanceModule(SteeringModule):
    weight = 0.9

    def __init__(self, neighbor_radius: float = 100.0):
        self.neighbor_radius = neighbor_radius

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        steering = pygame.math.Vector2(0, 0)

        # Hard collision avoidance: all registered objects (obstacles + ships at close range)
        for obj in registry.objects:
            if obj is ship:
                continue
            to_obj = obj.position - ship.position
            dist = to_obj.length()
            combined = ship.size + obj.size
            hard_r = combined
            soft_r = combined * 3.0
            if dist > soft_r or dist < 1e-6:
                continue
            if dist < hard_r:
                strength = 1.0
            else:
                strength = 1.0 - (dist - hard_r) / (soft_r - hard_r)
            steering -= safe_normalize(to_obj) * strength

        # Soft ship avoidance: allies and enemies within neighbor_radius,
        # at 1/3 of the SeparationModule strength formula
        for other in registry.ships:
            if other is ship:
                continue
            to_other = other.position - ship.position
            dist = to_other.length()
            if dist > self.neighbor_radius or dist < 1e-6:
                continue
            size_factor = other.size / max(ship.size, 0.01)
            strength = (1.0 - dist / self.neighbor_radius) * size_factor / 3.0
            steering -= safe_normalize(to_other) * strength

        # Cap so many-object clusters don't overwhelm approach modules
        if steering.length() > 1.0:
            steering = safe_normalize(steering)
        return steering
