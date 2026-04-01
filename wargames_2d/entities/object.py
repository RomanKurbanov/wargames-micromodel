import pygame
from registry import ShipRegistry


class Object:
    def __init__(self, position: pygame.math.Vector2, size: float = 10.0):
        self.position = pygame.math.Vector2(position)
        self.size = size
        ShipRegistry.register_object(self)

    def destroy(self):
        ShipRegistry.unregister_object(self)
