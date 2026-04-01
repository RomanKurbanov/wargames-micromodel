COLORS = [
    (0, 200, 0),    # 0 green
    (200, 0, 0),    # 1 red
    (0, 0, 200),    # 2 blue
    (200, 200, 0),  # 3 yellow
]


def get_color(faction_id: int) -> tuple:
    return COLORS[faction_id % len(COLORS)]
