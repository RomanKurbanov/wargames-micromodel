import pygame
from behavior.flock.friendly_base_module import FriendlyBaseModule
from behavior.steering_module import safe_normalize


class AlignmentModule(FriendlyBaseModule):
    weight = 0.3

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        neighbors = self.get_neighbors(ship, registry)
        # Only align with neighbors that have a loaded weapon so that retreating
        # reloaders don't drag the whole formation rearward.
        active = [n for n in neighbors if n.weapon is None or not n.weapon.reloading]
        if not active:
            return pygame.math.Vector2(0, 0)

        avg_vel = pygame.math.Vector2(0, 0)
        for n in active:
            avg_vel += n.velocity
        avg_vel /= len(active)

        return safe_normalize(avg_vel)
