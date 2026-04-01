import pygame


class Objective:
    """
    A shared team goal. Two variants:

    Area   — orbit a fixed world position (supply depot, control point, etc.)
             Construct with: Objective(position=Vector2(...), patrol_radius=200)

    Target — hunt a specific enemy ship; patrol_radius sets the engagement ring
             around it. Falls back to the last known position if the ship dies.
             Construct with: Objective(target_ship=some_ship, patrol_radius=150)
    """

    def __init__(
        self,
        position=None,
        target_ship=None,
        patrol_radius: float = 200.0,
    ):
        self._static_pos = pygame.math.Vector2(position) if position is not None else None
        self.target_ship = target_ship
        self.patrol_radius = patrol_radius

    @property
    def position(self):
        """Current world-space center. Returns None if the objective is gone."""
        if self.target_ship is not None:
            if self.target_ship.is_alive():
                return pygame.math.Vector2(self.target_ship.position)
            # Target dead — fall back to where it last was
            if self._static_pos is not None:
                return pygame.math.Vector2(self._static_pos)
            return None
        if self._static_pos is not None:
            return pygame.math.Vector2(self._static_pos)
        return None

    def is_valid(self) -> bool:
        return self.position is not None
