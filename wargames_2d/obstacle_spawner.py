import math
import random
import pygame

from entities.obstacle import Obstacle

# Ship size used in the spacing formula
_SHIP_SIZE = 10.0


def _min_spacing(size_a: float, size_b: float) -> float:
    """Minimum center-to-center distance so the CollisionAvoidance soft zones
    of two obstacles never overlap for any boid passing between them.

    soft_r(obj) = (ship_size + obj.size) * 3
    Zones don't overlap when:  d > soft_r_A + soft_r_B
    """
    return 3.0 * (2 * _SHIP_SIZE + size_a + size_b)


def spawn_obstacles(
    count: int = 14,
    min_size: float = 60.0,
    max_size: float = 110.0,
    world_half_x: float = 1400.0,
    world_half_y: float = 800.0,
    spawn_centers=None,
    spawn_clearance: float = 300.0,
    max_attempts: int = 20_000,
    seed: int = None,
) -> list:
    """
    Place `count` obstacles with guaranteed steering-independence spacing:
    no boid can be within the CollisionAvoidance soft zone of two obstacles
    simultaneously.

    spawn_centers: list of Vector2 positions to keep clear (faction spawns).
    spawn_clearance: min distance from each spawn center to any obstacle edge.
    """
    if seed is not None:
        random.seed(seed)
    if spawn_centers is None:
        spawn_centers = []

    placed: list[tuple[pygame.math.Vector2, float]] = []  # (pos, size)
    obstacles = []

    attempts = 0
    while len(obstacles) < count and attempts < max_attempts:
        attempts += 1

        size = random.uniform(min_size, max_size)
        pos = pygame.math.Vector2(
            random.uniform(-world_half_x, world_half_x),
            random.uniform(-world_half_y, world_half_y),
        )

        # Keep clear of faction spawn clusters
        if any(
            (pos - sc).length() < spawn_clearance + size
            for sc in spawn_centers
        ):
            continue

        # Enforce spacing rule against every already-placed obstacle
        if any(
            (pos - p).length() < _min_spacing(size, s)
            for p, s in placed
        ):
            continue

        obs = Obstacle(pos, size)
        obstacles.append(obs)
        placed.append((pos, size))

    return obstacles
