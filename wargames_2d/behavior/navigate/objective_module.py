import pygame
from behavior.steering_module import SteeringModule, safe_normalize


class ObjectiveModule(SteeringModule):
    weight = 0.8

    def __init__(self, inner_radius: float = 80.0):
        # inner_radius: fallback patrol radius when a raw Vector2 objective is used
        self.inner_radius = inner_radius

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        if ship.objective is None:
            return pygame.math.Vector2(0, 0)

        # Accept either an Objective object or a legacy raw Vector2
        if hasattr(ship.objective, 'position'):
            if not ship.objective.is_valid():
                return pygame.math.Vector2(0, 0)
            obj_pos = ship.objective.position
            patrol_r = ship.objective.patrol_radius
        else:
            obj_pos = ship.objective
            patrol_r = self.inner_radius

        to_obj = obj_pos - ship.position
        dist = to_obj.length()
        if dist < 1e-4:
            return pygame.math.Vector2(0, 0)

        direction = to_obj / dist                                    # toward center
        tangent = pygame.math.Vector2(-direction.y, direction.x)    # CCW tangent

        # Transition band: same width as patrol ring.  Ships are in full radial
        # approach at any distance beyond patrol_r + transition_band — no outer
        # cutoff so the objective always exerts pull no matter how far away.
        transition_band = patrol_r

        # Zone 1 — approach (any distance beyond transition edge): full radial pull
        if dist > patrol_r + transition_band:
            return direction

        # Zone 2 — transition: blend radial → tangential as ship nears the ring.
        # t=1 at outer edge (pure radial), t=0 at ring (pure tangential).
        if dist > patrol_r:
            t = (dist - patrol_r) / transition_band
            return safe_normalize(direction * t + tangent * (1.0 - t))

        # Zone 3 — inside ring: tangential orbit + outward nudge.
        # Nudge fades to zero at the ring edge so the boundary is continuous.
        inner_t = dist / max(patrol_r, 1e-4)   # 0 = center, 1 = ring edge
        outward = -direction
        return safe_normalize(tangent + outward * (1.0 - inner_t) * 0.5)
