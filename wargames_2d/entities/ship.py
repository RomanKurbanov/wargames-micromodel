import math
import pygame
from entities.object import Object
from entities.weapon import _los_blocked_any
from registry import ShipRegistry

_TARGET_SCAN_INTERVAL = 10   # frames between full target rescans (~0.17 s at 60 fps)


class Ship(Object):
    def __init__(self, position, faction: int = 0, size: float = 10.0):
        super().__init__(position, size)

        # Movement stats
        self.max_speed = 100.0
        self.forward_acceleration = 40.0
        self.brake_acceleration = 200.0
        self.turn_rate = 180.0      # deg/sec
        self.lateral_damping = 3.0

        # Combat state
        self.hp = 100.0
        self.max_hp = 100.0          # set by spawner; used for hp_frac normalisation
        self.shield = 200.0
        self.max_shield = 200.0      # set equal to initial shield; used for normalization
        self.shield_regen = 0.0     # shield points recovered per second
        self.faction = faction

        # Navigation
        self.target = None
        self.objective = None           # Vector2 or None
        self.visual_range = 800.0

        # Physics state
        self.velocity = pygame.math.Vector2(0, 0)
        self.heading_angle = 0.0        # degrees, 0 = +x axis
        self.course_lock = pygame.math.Vector2(0, 0)  # stub, always zero

        # Assigned externally by spawner
        self.brain = None
        self.weapon = None

        self._target_scan_cooldown: int = 0   # frames until next full scan

        ShipRegistry.register_ship(self)

    def is_alive(self) -> bool:
        return self.hp > 0

    def apply_damage(self, damage: float):
        if self.shield > 0:
            self.shield -= damage
            if self.shield < 0:
                self.hp += self.shield   # absorb overflow
                self.shield = 0
        else:
            self.hp -= damage
        if self.hp <= 0:
            self.hp = 0
            self.destroy()

    def set_target(self, target):
        self.target = target
        if self.weapon is not None:
            self.weapon.target = target

    def update_target(self, registry):
        # Fast path: hold current target while cooldown is active.
        if self._target_scan_cooldown > 0:
            self._target_scan_cooldown -= 1
            if self.target is not None and self.target.is_alive():
                if (self.target.position - self.position).length() < self.visual_range:
                    return
            # Target lost or left visual range — fall through to a rescan now.

        self._target_scan_cooldown = _TARGET_SCAN_INTERVAL

        fire_range   = self.weapon.range if self.weapon is not None else 0.0
        can_fire_at  = []
        in_fire_range = []
        closest      = None
        closest_dist = float('inf')

        # Use precomputed numpy arrays for static-obstacle LoS — much faster than
        # iterating registry.objects per candidate.  Ships are intentionally excluded
        # from LoS blocking here; weapon.can_fire_at still checks all objects at fire time.
        obs_pos = registry.obstacle_pos    # (N, 2) float64
        obs_r   = registry.obstacle_radii  # (N,)   float64
        p1x = self.position.x
        p1y = self.position.y

        for s in registry.ships:
            if s is self or s.faction == self.faction:
                continue
            d = (s.position - self.position).length()
            if d >= self.visual_range:
                continue
            if _los_blocked_any(p1x, p1y, s.position.x, s.position.y, obs_pos, obs_r):
                continue
            if d < closest_dist:
                closest_dist = d
                closest = s
            if d <= fire_range:
                in_fire_range.append(s)
                if self.weapon is not None and self.weapon.can_fire_at(self, s.position, registry):
                    can_fire_at.append(s)

        if can_fire_at:
            # Prefer targets we can shoot right now: focus the weakest
            best = min(can_fire_at, key=lambda s: s.hp + s.shield)
        elif in_fire_range:
            # Within range distance but not yet aligned: approach the weakest to align
            best = min(in_fire_range, key=lambda s: s.hp + s.shield)
        else:
            best = closest
        self.set_target(best)

    def get_heading_vector(self) -> pygame.math.Vector2:
        rad = math.radians(self.heading_angle)
        return pygame.math.Vector2(math.cos(rad), math.sin(rad))

    def update(self, dt: float):
        """Per-frame update: regenerate shield up to max_shield."""
        if self.shield_regen > 0 and self.shield < self.max_shield:
            self.shield = min(self.max_shield, self.shield + self.shield_regen * dt)

    def destroy(self):
        self.hp = 0          # is_alive() is always False after destroy()
        ShipRegistry.unregister_ship(self)
        super().destroy()
