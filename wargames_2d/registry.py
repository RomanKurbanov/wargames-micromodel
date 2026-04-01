class ShipRegistry:
    objects = []
    ships = []
    ships_by_faction = {}

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
    def clear():
        ShipRegistry.objects.clear()
        ShipRegistry.ships.clear()
        ShipRegistry.ships_by_faction.clear()
