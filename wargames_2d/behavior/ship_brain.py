import math
import pygame
from behavior.steering_module import safe_normalize
from behavior.personality import build_ship_state


def _move_towards(current: float, target: float, max_delta: float) -> float:
    diff = target - current
    if abs(diff) <= max_delta:
        return target
    return current + math.copysign(max_delta, diff)


TICK_INTERVAL = 0.25   # seconds between weight refreshes


class ShipBrain:
    def __init__(self, modules=None, personality=None, tick_interval=TICK_INTERVAL):
        self.modules = modules or []
        self.personality = personality   # Personality | None
        self.tick_interval = tick_interval
        self._tick_timer = tick_interval  # start at max so first frame fires immediately
        self.overrides: dict = {}         # last computed weight overrides (public, read-only)

    def update(self, ship, dt: float, registry):
        self._tick_timer += dt
        if self._tick_timer >= self.tick_interval:
            self._tick_timer -= self.tick_interval
            self.overrides = self.compute_weight_overrides(ship, registry)

        steering = pygame.math.Vector2(0, 0)
        for mod in self.modules:
            w = self.overrides.get(type(mod).__name__, mod.weight)
            # Negative weights invert a module's direction (semantically wrong);
            # NaN/Inf corrupt heading_angle permanently via atan2.
            # Floor at zero: suppressed module, not reversed one.
            if not math.isfinite(w):
                continue
            w = max(0.0, w)
            if w > 0.0:
                steering += mod.compute_steering(ship, registry) * w
        self._aviate(ship, steering, dt)

    def compute_weight_overrides(self, ship, registry) -> dict:
        """
        Delegates to the injected Personality if present.
        The personality receives an immutable ShipState snapshot — it
        cannot touch live game objects.
        Returns {} (all static weights) when no personality is set.
        """
        if self.personality is None:
            return {}
        state = build_ship_state(ship, registry)
        return self.personality.get_weights(state)

    def _aviate(self, ship, steering: pygame.math.Vector2, dt: float):
        steering_len = steering.length()
        desire = min(steering_len, 1.0)

        # Desired movement direction
        if steering_len > 1e-4:
            desired_move_dir = steering / steering_len
        elif ship.velocity.length() > 1e-4:
            desired_move_dir = safe_normalize(ship.velocity)
        else:
            desired_move_dir = ship.get_heading_vector()

        target_speed = desire * ship.max_speed

        # course_lock is always zero (stub) — face desired move direction
        if ship.course_lock.length() > 1e-4:
            facing_target = safe_normalize(ship.course_lock)
        else:
            facing_target = desired_move_dir

        # Rotate heading toward facing_target by at most turn_rate * dt
        facing_angle = math.degrees(math.atan2(facing_target.y, facing_target.x))
        diff = (facing_angle - ship.heading_angle + 180) % 360 - 180
        max_turn = ship.turn_rate * dt
        if abs(diff) <= max_turn:
            ship.heading_angle = facing_angle
        else:
            ship.heading_angle += math.copysign(max_turn, diff)
        ship.heading_angle %= 360

        # Forward unit vector
        heading_rad = math.radians(ship.heading_angle)
        forward = pygame.math.Vector2(math.cos(heading_rad), math.sin(heading_rad))

        # Decompose velocity into forward and lateral components
        forward_speed = ship.velocity.dot(forward)
        lateral_vel = ship.velocity - forward * forward_speed

        # Lateral damping (exponential decay)
        damp_factor = 1.0 - math.exp(-ship.lateral_damping * dt)
        lateral_vel = lateral_vel * (1.0 - damp_factor)

        # Accelerate or brake toward target_speed
        if forward_speed < target_speed:
            new_forward = _move_towards(forward_speed, target_speed, ship.forward_acceleration * dt)
        else:
            new_forward = _move_towards(forward_speed, target_speed, ship.brake_acceleration * dt)

        # Deadzone
        if abs(new_forward) < 0.05 and target_speed < 0.05:
            new_forward = 0.0

        ship.velocity = forward * new_forward + lateral_vel

        # Clamp to max_speed
        spd = ship.velocity.length()
        if spd > ship.max_speed:
            ship.velocity *= ship.max_speed / spd

        ship.position += ship.velocity * dt
