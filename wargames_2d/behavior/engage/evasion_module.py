import pygame
from behavior.engage.enemy_base_module import EnemyBaseModule
from behavior.steering_module import safe_normalize


class EvasionModule(EnemyBaseModule):
    weight = 0.6

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        # Evade the current target; fall back to closest in-range enemy
        enemy = ship.target if (ship.target and ship.target.is_alive()) else None
        if enemy is None:
            enemies = self.get_enemies(ship, registry)
            if not enemies:
                return pygame.math.Vector2(0, 0)
            enemy = min(enemies, key=lambda e: (e.position - ship.position).length())

        if enemy.weapon is None:
            return pygame.math.Vector2(0, 0)

        to_enemy = enemy.position - ship.position
        dist = to_enemy.length()
        if dist < 1e-6:
            return pygame.math.Vector2(0, 0)

        hard_r = enemy.weapon.range
        soft_r = hard_r * 1.5

        if dist > soft_r:
            return pygame.math.Vector2(0, 0)

        # Closing speed: positive when approaching enemy
        rel_vel = ship.velocity - enemy.velocity
        closing_speed = rel_vel.dot(to_enemy / dist)
        approach_factor = max(0.0, min(closing_speed / max(ship.max_speed, 1.0), 1.0))

        if dist < hard_r:
            strength = 1.0 + approach_factor * 0.5
        else:
            t = (dist - hard_r) / (soft_r - hard_r)
            strength = (1.0 - t) * (1.0 + approach_factor * 0.5)

        return -(to_enemy / dist) * strength
