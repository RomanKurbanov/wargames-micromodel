import pygame
from behavior.flock.friendly_base_module import FriendlyBaseModule
from behavior.steering_module import safe_normalize


class CohesionModule(FriendlyBaseModule):
    weight = 0.15

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        neighbors = self.get_neighbors(ship, registry)
        if not neighbors:
            return pygame.math.Vector2(0, 0)

        center = pygame.math.Vector2(0, 0)
        total_w = 0.0

        for n in neighbors:
            to_n = n.position - ship.position
            dist = to_n.length()

            # Approach speed: does the neighbor's velocity bring it toward us?
            approach = 0.0
            nspd = n.velocity.length()
            if nspd > 1e-4 and dist > 1e-4:
                approach_dir = safe_normalize(ship.position - n.position)
                approach = max(0.0, safe_normalize(n.velocity).dot(approach_dir))

            size_factor = n.size / max(ship.size, 0.01)
            dist_factor = max(1.0 - dist / self.neighbor_radius, 0.01)

            w = (1.0 + approach) * size_factor * dist_factor
            center += n.position * w
            total_w += w

        if total_w < 1e-6:
            return pygame.math.Vector2(0, 0)

        center /= total_w
        return safe_normalize(center - ship.position)
