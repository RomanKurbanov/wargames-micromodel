import pygame
from entities.object import Object


class Obstacle(Object):
    """Static world obstacle. Auto-registers in ShipRegistry.objects so
    CollisionAvoidanceModule and weapon LoS checks pick it up for free."""

    COLOR = (75, 65, 55)
    OUTLINE_COLOR = (110, 95, 80)

    def __init__(self, position: pygame.math.Vector2, size: float = 35.0):
        super().__init__(position, size)
