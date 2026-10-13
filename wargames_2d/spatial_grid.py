class SpatialGrid:
    """
    Uniform-grid spatial hash for fast radius queries.

    Rebuilt once per frame from the full ship list; queried by steering
    modules instead of iterating all ships.  cell_size=200 keeps the scan
    window to 9 cells for radii up to 200 px and 49 cells for radii up to
    600 px, vs. a full O(N) list walk for every ship every frame.
    """

    def __init__(self, cell_size: float = 200.0):
        self.cell_size = cell_size
        self._grid: dict = {}   # (ix, iy) → list[entity]

    def build(self, entities) -> None:
        self._grid.clear()
        cs  = self.cell_size
        get = self._grid.get
        for e in entities:
            k = (int(e.position.x / cs), int(e.position.y / cs))
            bucket = get(k)
            if bucket is None:
                self._grid[k] = [e]
            else:
                bucket.append(e)

    def query(self, x: float, y: float, radius: float) -> list:
        """Return all entities whose grid cell overlaps the circle (x, y, radius)."""
        cs  = self.cell_size
        r   = int(radius / cs) + 1
        cx  = int(x / cs)
        cy  = int(y / cs)
        get = self._grid.get
        result = []
        for ix in range(cx - r, cx + r + 1):
            for iy in range(cy - r, cy + r + 1):
                bucket = get((ix, iy))
                if bucket:
                    result.extend(bucket)
        return result
