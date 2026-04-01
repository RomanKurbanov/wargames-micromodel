import pygame
from behavior.engage.enemy_base_module import EnemyBaseModule
from behavior.steering_module import safe_normalize


class FiringSolutionModule(EnemyBaseModule):
    weight = 0.8

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        if ship.target is None or not ship.target.is_alive():
            return pygame.math.Vector2(0, 0)
        if ship.weapon is None:
            return pygame.math.Vector2(0, 0)

        # Already in firing position — no steering nudge needed
        if ship.weapon.can_fire_at(ship, ship.target.position, registry):
            return pygame.math.Vector2(0, 0)

        # Move toward target
        to_target = ship.target.position - ship.position
        return safe_normalize(to_target)
