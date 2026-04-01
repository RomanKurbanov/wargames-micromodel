from abc import ABC, abstractmethod
import pygame


def safe_normalize(v: pygame.math.Vector2) -> pygame.math.Vector2:
    """Return normalized v, or zero vector if v is near-zero."""
    l = v.length()
    if l < 1e-6:
        return pygame.math.Vector2(0, 0)
    return v / l


class SteeringModule(ABC):
    weight: float = 0.5

    @abstractmethod
    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        ...
