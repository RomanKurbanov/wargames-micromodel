import numpy as np
from spatial_grid import SpatialGrid


class ShipRegistry:
    objects = []
    ships = []
    ships_by_faction = {}
    obstacles: list = []                                    # Obstacle objects only (static)
    # Precomputed numpy arrays for static-obstacle LoS (populated by setup_scenario)
    obstacle_pos    = np.zeros((0, 2), dtype=np.float64)   # (N, 2) x/y
    obstacle_radii  = np.zeros(0,      dtype=np.float64)   # (N,)
    # Spatial grid over all ships — rebuilt once per frame before brain updates
    ships_grid: SpatialGrid = SpatialGrid(cell_size=200.0)

    @staticmethod
    def register_object(obj):
        ShipRegistry.objects.append(obj)

    @staticmethod
    def unregister_object(obj):
        if obj in ShipRegistry.objects:
            ShipRegistry.objects.remove(obj)

    @staticmethod
    def register_ship(ship):
        ShipRegistry.ships.append(ship)
        fid = ship.faction
        if fid not in ShipRegistry.ships_by_faction:
            ShipRegistry.ships_by_faction[fid] = []
        ShipRegistry.ships_by_faction[fid].append(ship)

    @staticmethod
    def unregister_ship(ship):
        if ship in ShipRegistry.ships:
            ShipRegistry.ships.remove(ship)
        fid = ship.faction
        lst = ShipRegistry.ships_by_faction.get(fid)
        if lst and ship in lst:
            lst.remove(ship)

    @staticmethod
    def rebuild_ships_grid() -> None:
        """Rebuild the spatial grid from the current live ship list. Call once per frame."""
        ShipRegistry.ships_grid.build(ShipRegistry.ships)

    @staticmethod
    def rebuild_obstacle_cache(obstacles: list) -> None:
        """Store obstacle objects and build numpy LoS arrays. Call once after spawning."""
        ShipRegistry.obstacles = list(obstacles)
        if obstacles:
            ShipRegistry.obstacle_pos = np.array(
                [(o.position.x, o.position.y) for o in obstacles], dtype=np.float64
            )
            ShipRegistry.obstacle_radii = np.array(
                [o.size for o in obstacles], dtype=np.float64
            )
        else:
            ShipRegistry.obstacle_pos   = np.zeros((0, 2), dtype=np.float64)
            ShipRegistry.obstacle_radii = np.zeros(0,      dtype=np.float64)

    @staticmethod
    def clear():
        ShipRegistry.objects.clear()
        ShipRegistry.ships.clear()
        ShipRegistry.ships_by_faction.clear()
        ShipRegistry.obstacles      = []
        ShipRegistry.obstacle_pos   = np.zeros((0, 2), dtype=np.float64)
        ShipRegistry.obstacle_radii = np.zeros(0,      dtype=np.float64)
        ShipRegistry.ships_grid     = SpatialGrid(cell_size=200.0)
