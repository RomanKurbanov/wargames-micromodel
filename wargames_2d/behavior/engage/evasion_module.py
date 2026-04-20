import pygame
from behavior.engage.enemy_base_module import EnemyBaseModule
from behavior.steering_module import safe_normalize


class EvasionModule(EnemyBaseModule):
    weight = 0.6

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        enemies = self.get_enemies(ship, registry)
        if not enemies:
            return pygame.math.Vector2(0, 0)

        result = pygame.math.Vector2(0, 0)

        for enemy in enemies:
            if enemy.weapon is None:
                continue

            to_enemy = enemy.position - ship.position
            dist = to_enemy.length()
            if dist < 1e-6:
                continue

            hard_r = enemy.weapon.range
            soft_r = hard_r * 1.5

            if dist > soft_r:
                continue

            # Closing speed: positive when approaching enemy
            rel_vel = ship.velocity - enemy.velocity
            closing_speed = rel_vel.dot(to_enemy / dist)
            approach_factor = max(0.0, min(closing_speed / max(ship.max_speed, 1.0), 1.0))

            if dist < hard_r:
                strength = 1.0 + approach_factor * 0.5
            else:
                t = (dist - hard_r) / (soft_r - hard_r)
                strength = (1.0 - t) * (1.0 + approach_factor * 0.5)

            result += -(to_enemy / dist) * strength

        # Clamp to the same magnitude ceiling a single enemy could produce,
        # so multiple enemies shift direction without amplifying throttle.
        mag = result.length()
        if mag > 1.5:
            result = result / mag * 1.5

        return result
