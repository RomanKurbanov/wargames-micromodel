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

        to_target = ship.target.position - ship.position
        dist = to_target.length()
        if dist < 1e-4:
            return pygame.math.Vector2(0, 0)

        # In range AND can fire — request a facing lock toward target via course_lock.
        # ShipBrain scales course_lock by the effective AttackModule weight before _aviate.
        # Return zero so movement comes entirely from other modules (evasion, objective, etc.).
        if dist <= ship.weapon.range and ship.weapon.can_fire_at(ship, ship.target.position, registry):
            ship.course_lock = to_target / dist  # unit vector toward target
            return pygame.math.Vector2(0, 0)

        # Outside range or not yet aligned — close the distance / establish firing solution
        return to_target / dist
