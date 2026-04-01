"""
Scenario definition and loading.

A scenario file is a JSON document that fully describes one simulation
run: factions, their ships and weapons, objectives, obstacles, and output
settings.  This module contains only data classes and a JSON parser —
no game objects, no pygame.

Load a scenario:
    from scenario import load_scenario
    config = load_scenario("scenarios/default.json")
"""

import json
import os
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Config dataclasses — one per JSON section
# ---------------------------------------------------------------------------

@dataclass
class ShipConfig:
    """Per-faction ship stat overrides.  Missing fields keep engine defaults."""
    max_speed:             float = 100.0
    forward_acceleration:  float = 40.0
    brake_acceleration:    float = 200.0
    turn_rate:             float = 180.0   # deg/sec
    lateral_damping:       float = 3.0
    hp:                    float = 100.0
    shield:                float = 200.0
    shield_regen:          float = 0.0     # shield points per second
    size:                  float = 10.0    # render radius / collision radius
    visual_range:          float = 800.0


@dataclass
class WeaponConfig:
    """Per-faction weapon stat overrides."""
    range:           float = 400.0
    damage:          float = 10.0
    fire_rate:       float = 5.0    # shots/sec within a cassette
    max_fire_angle:  float = 25.0   # half-cone degrees
    cassette_size:   int   = 5
    reload_duration: float = 6.0    # seconds


@dataclass
class ObjectiveConfig:
    """Navigation goal for one faction.  position=None means no objective."""
    position:      Optional[tuple] = None  # (x, y) world coords
    patrol_radius: float           = 200.0


@dataclass
class FactionConfig:
    id:            int
    ship_count:    int
    spawn_center:  tuple          # (x, y)
    spawn_radius:  float          = 150.0
    objective:     ObjectiveConfig = field(default_factory=ObjectiveConfig)
    ship:          ShipConfig      = field(default_factory=ShipConfig)
    weapon:        WeaponConfig    = field(default_factory=WeaponConfig)
    # 'default' | 'none' | 'ml:path/to/model.npz'
    personality:   str             = 'default'


@dataclass
class ObstacleConfig:
    enabled:         bool          = True
    count:           int           = 14
    min_size:        float         = 60.0
    max_size:        float         = 110.0
    world_half_x:    float         = 1400.0
    world_half_y:    float         = 800.0
    spawn_clearance: float         = 300.0  # clear radius around each spawn center
    seed:            Optional[int] = None   # None = random each run


@dataclass
class OutputConfig:
    """Controls result data written to disk after a headless run."""
    file:            Optional[str] = None  # path for JSON results; None = no file
    record_interval: int           = 60    # steps between timeline snapshots (0 = off)


@dataclass
class SimulationConfig:
    max_steps: Optional[int] = None   # None = run until one side is eliminated
    headless:  bool          = False
    output:    OutputConfig  = field(default_factory=OutputConfig)


@dataclass
class ScenarioConfig:
    name:         str
    description:  str
    world_bounds: float            # BoundsModule radius for all ships
    factions:     list             # list[FactionConfig]
    obstacles:    ObstacleConfig
    simulation:   SimulationConfig


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def _parse_ship(d: dict) -> ShipConfig:
    cfg = ShipConfig()
    for key in ('max_speed', 'forward_acceleration', 'brake_acceleration',
                'turn_rate', 'lateral_damping', 'hp', 'shield', 'shield_regen',
                'size', 'visual_range'):
        if key in d:
            setattr(cfg, key, float(d[key]))
    return cfg


def _parse_weapon(d: dict) -> WeaponConfig:
    cfg = WeaponConfig()
    for key in ('range', 'damage', 'fire_rate', 'max_fire_angle', 'reload_duration'):
        if key in d:
            setattr(cfg, key, float(d[key]))
    if 'cassette_size' in d:
        cfg.cassette_size = int(d['cassette_size'])
    return cfg


def _parse_objective(d: dict) -> ObjectiveConfig:
    raw_pos = d.get('position')
    return ObjectiveConfig(
        position      = (float(raw_pos[0]), float(raw_pos[1])) if raw_pos else None,
        patrol_radius = float(d.get('patrol_radius', 200.0)),
    )


def _parse_faction(d: dict) -> FactionConfig:
    raw_center = d['spawn_center']
    return FactionConfig(
        id           = int(d['id']),
        ship_count   = int(d['ship_count']),
        spawn_center = (float(raw_center[0]), float(raw_center[1])),
        spawn_radius = float(d.get('spawn_radius', 150.0)),
        objective    = _parse_objective(d.get('objective', {})),
        ship         = _parse_ship(d.get('ship', {})),
        weapon       = _parse_weapon(d.get('weapon', {})),
        personality  = d.get('personality', 'default'),
    )


def _parse_obstacles(d: dict) -> ObstacleConfig:
    cfg = ObstacleConfig()
    bool_keys  = ('enabled',)
    int_keys   = ('count',)
    float_keys = ('min_size', 'max_size', 'world_half_x', 'world_half_y', 'spawn_clearance')
    for k in bool_keys:
        if k in d: setattr(cfg, k, bool(d[k]))
    for k in int_keys:
        if k in d: setattr(cfg, k, int(d[k]))
    for k in float_keys:
        if k in d: setattr(cfg, k, float(d[k]))
    if 'seed' in d:
        cfg.seed = int(d['seed']) if d['seed'] is not None else None
    return cfg


def _parse_output(d: dict) -> OutputConfig:
    return OutputConfig(
        file            = d.get('file'),
        record_interval = int(d.get('record_interval', 60)),
    )


def _parse_simulation(d: dict) -> SimulationConfig:
    raw_max = d.get('max_steps')
    return SimulationConfig(
        max_steps = int(raw_max) if raw_max is not None else None,
        headless  = bool(d.get('headless', False)),
        output    = _parse_output(d.get('output', {})),
    )


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> ScenarioConfig:
    """
    Load and parse a scenario JSON file.
    Raises FileNotFoundError if the path does not exist.
    Raises ValueError for structural problems.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path, encoding='utf-8') as f:
        raw = json.load(f)

    factions = [_parse_faction(fd) for fd in raw.get('factions', [])]
    if not factions:
        raise ValueError(f"{path}: must define at least one faction")

    ids = [fc.id for fc in factions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: faction ids must be unique")

    return ScenarioConfig(
        name         = raw.get('name', 'Unnamed Scenario'),
        description  = raw.get('description', ''),
        world_bounds = float(raw.get('world_bounds', 2000.0)),
        factions     = factions,
        obstacles    = _parse_obstacles(raw.get('obstacles', {})),
        simulation   = _parse_simulation(raw.get('simulation', {})),
    )
