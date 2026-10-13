import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class CollisionAvoidanceModule(SteeringModule):
    weight = 0.9

    def __init__(self, neighbor_radius: float = 100.0):
        self.neighbor_radius = neighbor_radius

    # Maximum possible ship-ship soft_r = (max_ship_size + max_ship_size) * 3.
    # Covers hard collision for any ship pair; grid cell_size=200 keeps this at r=1.
    _SHIP_COLLISION_R: float = 90.0

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        steering = pygame.math.Vector2(0, 0)
        px, py   = ship.position.x, ship.position.y
        ss       = ship.size

        # Static obstacle avoidance — 14 obstacles, iterate directly.
        for obj in registry.obstacles:
            to_obj   = obj.position - ship.position
            dist     = to_obj.length()
            combined = ss + obj.size
            soft_r   = combined * 3.0
            if dist > soft_r or dist < 1e-6:
                continue
            hard_r   = combined
            strength = 1.0 if dist < hard_r else 1.0 - (dist - hard_r) / (soft_r - hard_r)
            steering -= safe_normalize(to_obj) * strength

        # Ship avoidance via spatial grid (hard collision + soft repulsion).
        # Uses the largest possible ship-ship soft_r as the query radius.
        r2 = self._SHIP_COLLISION_R ** 2
        nr = self.neighbor_radius
        for other in registry.ships_grid.query(px, py, self._SHIP_COLLISION_R):
            if other is ship:
                continue
            dx   = other.position.x - px
            dy   = other.position.y - py
            d2   = dx * dx + dy * dy
            if d2 < 1e-12:
                continue
            dist     = d2 ** 0.5
            combined = ss + other.size
            soft_r   = combined * 3.0
            if d2 <= soft_r * soft_r:
                # Hard collision repulsion
                hard_r   = combined
                strength = 1.0 if dist < hard_r else 1.0 - (dist - hard_r) / (soft_r - hard_r)
                steering -= safe_normalize(other.position - ship.position) * strength
            if d2 <= nr * nr:
                # Soft separation at 1/3 strength
                size_factor = other.size / max(ss, 0.01)
                steering   -= safe_normalize(other.position - ship.position) * (1.0 - dist / nr) * size_factor / 3.0

        # Cap so many-object clusters don't overwhelm approach modules
        if steering.length() > 1.0:
            steering = safe_normalize(steering)
        return steering
