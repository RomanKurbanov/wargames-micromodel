import math
import numpy as np
import pygame


def _los_blocked_any(
    p1x: float, p1y: float,
    p2x: float, p2y: float,
    centers: np.ndarray,
    radii: np.ndarray,
) -> bool:
    """
    Return True if segment p1→p2 is blocked by any of the circles.
    centers: float64 array of shape (N, 2)
    radii:   float64 array of shape (N,)
    Faster than N calls to _segment_circle_intersects for N > ~4.
    """
    if len(radii) == 0:
        return False
    cx = centers[:, 0];  cy = centers[:, 1]
    dx = p2x - p1x;      dy = p2y - p1y
    a  = dx * dx + dy * dy
    if a < 1e-10:
        return False
    fx = p1x - cx;       fy = p1y - cy
    b    = 2.0 * (fx * dx + fy * dy)
    disc = b * b - 4.0 * a * (fx * fx + fy * fy - radii * radii)
    mask = disc >= 0.0
    if not np.any(mask):
        return False
    sq    = np.sqrt(disc[mask])
    bm    = b[mask]
    inv2a = 0.5 / a
    t1    = (-bm - sq) * inv2a
    t2    = (-bm + sq) * inv2a
    return bool(np.any(((t1 >= 0.0) & (t1 <= 1.0)) | ((t2 >= 0.0) & (t2 <= 1.0))))


def _segment_circle_intersects(
    p1: pygame.math.Vector2,
    p2: pygame.math.Vector2,
    center: pygame.math.Vector2,
    radius: float,
) -> bool:
    """True if segment p1→p2 intersects the circle (center, radius)."""
    d = p2 - p1
    f = p1 - center
    a = d.dot(d)
    if a < 1e-10:
        return False
    b = 2.0 * f.dot(d)
    c = f.dot(f) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc < 0:
        return False
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    return (0.0 <= t1 <= 1.0) or (0.0 <= t2 <= 1.0)


class Weapon:
    def __init__(self):
        self.range = 400.0
        self.damage = 10.0
        self.fire_rate = 5.0            # shots / sec within cassette
        self.max_fire_angle = 25.0      # degrees half-cone
        self.cassette_size = 5
        self.reload_duration = 6.0      # sec

        self.target = None
        self.owner = None

        # State
        self.shots_fired = 0
        self.shot_cooldown = 0.0
        self.reload_timer = 0.0
        self.reloading = False

        # (start_pos, end_pos, remaining_sec)
        self.tracers: list = []

    def initialize(self, owner):
        self.owner = owner

    def can_fire_at(self, ship, target_pos: pygame.math.Vector2, registry) -> bool:
        to_target = target_pos - ship.position
        dist = to_target.length()
        if dist > self.range:
            return False

        # Angle check
        heading = ship.get_heading_vector()
        if dist > 0.001:
            angle_diff = math.degrees(
                math.atan2(to_target.y, to_target.x)
                - math.atan2(heading.y, heading.x)
            )
            angle_diff = (angle_diff + 180) % 360 - 180
            if abs(angle_diff) > self.max_fire_angle:
                return False

        # Line-of-sight: blocked by any object except shooter and target
        target_ship = self.target
        for obj in registry.objects:
            if obj is ship or obj is target_ship:
                continue
            if _segment_circle_intersects(ship.position, target_pos, obj.position, obj.size):
                return False

        return True

    def update(self, ship, dt: float, registry):
        # Age tracers
        self.tracers = [(s, e, t - dt) for s, e, t in self.tracers if t - dt > 0]

        if self.reloading:
            self.reload_timer -= dt
            if self.reload_timer <= 0:
                self.reloading = False
                self.shots_fired = 0
                self.reload_timer = 0.0
            return

        if self.shot_cooldown > 0:
            self.shot_cooldown -= dt

        if self.target is None or not self.target.is_alive():
            return

        if self.shot_cooldown <= 0 and self.shots_fired < self.cassette_size:
            if self.can_fire_at(ship, self.target.position, registry):
                self._fire_single_shot(ship)

        if self.shots_fired >= self.cassette_size:
            self.reloading = True
            self.reload_timer = self.reload_duration

    def _fire_single_shot(self, ship):
        if self.target is None:
            return
        start = pygame.math.Vector2(ship.position)
        end = pygame.math.Vector2(self.target.position)
        self.target.apply_damage(self.damage)
        self.tracers.append((start, end, 0.08))
        self.shots_fired += 1
        self.shot_cooldown = 1.0 / self.fire_rate
