import pygame
from behavior.engage.enemy_base_module import EnemyBaseModule
from behavior.steering_module import safe_normalize


class EvasionModule(EnemyBaseModule):
    weight = 0.6

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        enemies = self.get_enemies(ship, registry)
        if not enemies:
            return pygame.math.Vector2(0, 0)

        spx = ship.position.x;  spy = ship.position.y
        svx = ship.velocity.x;  svy = ship.velocity.y
        inv_max_speed = 1.0 / (ship.max_speed if ship.max_speed > 1.0 else 1.0)

        wx = wy = total_weight = 0.0

        for enemy in enemies:
            if enemy.weapon is None:
                continue

            dx   = enemy.position.x - spx
            dy   = enemy.position.y - spy
            d2   = dx * dx + dy * dy
            if d2 < 1e-12:
                continue
            dist = d2 ** 0.5

            hard_r = enemy.weapon.range
            soft_r = hard_r * 1.5
            if dist > soft_r:
                continue

            # Closing speed: positive when approaching enemy (float-only)
            rel_vx = svx - enemy.velocity.x
            rel_vy = svy - enemy.velocity.y
            inv_d  = 1.0 / dist
            closing_speed   = (rel_vx * dx + rel_vy * dy) * inv_d
            approach_factor = max(0.0, min(closing_speed * inv_max_speed, 1.0))

            if dist < hard_r:
                strength = 1.0 + approach_factor * 0.5
            else:
                t        = (dist - hard_r) / (soft_r - hard_r)
                strength = (1.0 - t) * (1.0 + approach_factor * 0.5)

            # Weight by proximity so closer enemies dominate
            pw          = inv_d
            total_weight += pw
            scale        = -strength * pw * inv_d   # -(to_enemy / dist) * strength * pw
            wx          += dx * scale
            wy          += dy * scale

        if total_weight < 1e-9:
            return pygame.math.Vector2(0, 0)

        inv_tw = 1.0 / total_weight
        return pygame.math.Vector2(wx * inv_tw, wy * inv_tw)
