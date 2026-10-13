import math
import pygame
from behavior.flock.friendly_base_module import FriendlyBaseModule
from behavior.steering_module import safe_normalize


class CohesionModule(FriendlyBaseModule):
    weight = 0.15

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        neighbors = self.get_neighbors(ship, registry)
        if not neighbors:
            return pygame.math.Vector2(0, 0)

        spx = ship.position.x;  spy = ship.position.y
        nr  = self.neighbor_radius
        inv_ss = 1.0 / (ship.size if ship.size > 0.01 else 0.01)

        cx = cy = total_w = 0.0

        for n in neighbors:
            dx   = n.position.x - spx
            dy   = n.position.y - spy
            dist = (dx * dx + dy * dy) ** 0.5

            # Approach speed: how fast is the neighbor closing toward us (float-only)
            nvx  = n.velocity.x;  nvy = n.velocity.y
            nsp2 = nvx * nvx + nvy * nvy
            approach = 0.0
            if nsp2 > 1e-8 and dist > 1e-4:
                # dot(normalize(vel), normalize(ship - n)) = (-dx*nvx + -dy*nvy) / (dist * sqrt(nsp2))
                approach = max(0.0, (-nvx * dx - nvy * dy) / (dist * nsp2 ** 0.5))

            size_factor = n.size * inv_ss
            dist_factor = max(1.0 - dist / nr, 0.01)

            w       = (1.0 + approach) * size_factor * dist_factor
            cx     += n.position.x * w
            cy     += n.position.y * w
            total_w += w

        if total_w < 1e-6:
            return pygame.math.Vector2(0, 0)

        return safe_normalize(pygame.math.Vector2(cx / total_w - spx, cy / total_w - spy))
