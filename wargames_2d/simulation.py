"""
Scenario runner.

Provides two public functions:

    setup_scenario(config)  — initialise game state from a ScenarioConfig.
                              Used by both headless and display modes.

    run_scenario(config)    — run a full headless episode and return a
                              results dict that can be serialised to JSON.

The display loop lives in main.py.  This module has no pygame.display
dependency and can be imported in training scripts without opening a window.
"""

import datetime
import json
import os

import pygame

from registry import ShipRegistry
from scenario import ScenarioConfig, FactionConfig
from spawner import spawn_faction
from obstacle_spawner import spawn_obstacles
from objective import Objective


# ---------------------------------------------------------------------------
# Personality resolver
# ---------------------------------------------------------------------------

def _resolve_personality(spec: str):
    """Map a personality string from the scenario file to a Personality instance."""
    if spec == 'default':
        from behavior.personality import DefaultPersonality
        return DefaultPersonality()
    if spec == 'none':
        return None
    if spec == 'tactical':
        from behavior.personality.tactical_personality import TacticalPersonality
        return TacticalPersonality()
    if spec == 'tactical2':
        from behavior.personality.tactical_personality2 import TacticalPersonality2
        return TacticalPersonality2()
    if spec.startswith('ml:'):
        from behavior.personality.ml_personality import MLPersonality
        return MLPersonality(spec[3:])
    raise ValueError(
        f"Unknown personality spec {spec!r}. "
        f"Valid values: 'default', 'none', 'tactical', 'ml:<path/to/model.npz>'"
    )


# ---------------------------------------------------------------------------
# Shared setup — called by both headless runner and interactive main.py
# ---------------------------------------------------------------------------

def setup_scenario(config: ScenarioConfig) -> dict:
    """
    Clear the registry and spawn all game objects described by *config*.

    Returns a state dict with everything the render loop needs:
    {
        'obstacles':          list[Obstacle],
        'faction_objectives': {faction_id: Objective | None},
        'initial_counts':     {faction_id: int},
    }
    """
    ShipRegistry.clear()

    # --- Obstacles ---
    obs_cfg = config.obstacles
    spawn_centers = [pygame.math.Vector2(*fc.spawn_center) for fc in config.factions]

    obstacles = []
    if obs_cfg.enabled:
        obstacles = spawn_obstacles(
            count          = obs_cfg.count,
            min_size       = obs_cfg.min_size,
            max_size       = obs_cfg.max_size,
            world_half_x   = obs_cfg.world_half_x,
            world_half_y   = obs_cfg.world_half_y,
            spawn_centers  = spawn_centers,
            spawn_clearance= obs_cfg.spawn_clearance,
            seed           = obs_cfg.seed,
        )
    ShipRegistry.rebuild_obstacle_cache(obstacles)

    # --- Factions ---
    faction_objectives = {}
    initial_counts     = {}

    for fc in config.factions:
        center = pygame.math.Vector2(*fc.spawn_center)

        obj = None
        if fc.objective.position is not None:
            obj = Objective(
                position      = pygame.math.Vector2(*fc.objective.position),
                patrol_radius = fc.objective.patrol_radius,
            )
        faction_objectives[fc.id] = obj

        personality = _resolve_personality(fc.personality)

        ships = spawn_faction(
            faction      = fc.id,
            center       = center,
            objective    = obj,
            groups       = fc.groups,
            spawn_radius = fc.spawn_radius,
            personality  = personality,
        )
        initial_counts[fc.id] = len(ships)

    return {
        'obstacles':          obstacles,
        'faction_objectives': faction_objectives,
        'initial_counts':     initial_counts,
    }


# ---------------------------------------------------------------------------
# Batch ML inference helper
# ---------------------------------------------------------------------------

def _batch_ml_tick(snapshot: list, registry, dt: float) -> None:
    """
    For every alive ship whose brain holds an MLPersonality and whose tick
    timer will fire this frame, build state vectors in bulk, run one matrix
    multiply per personality instance (faction), and distribute the weight
    rows back.  ShipBrain.update() will see _batch_ticked=True and skip the
    redundant per-ship call.
    """
    try:
        from behavior.personality.ml_personality import MLPersonality
    except ImportError:
        return

    import numpy as np
    from behavior.personality import build_ship_state, state_to_vector

    # Group ships that are due for a tick by personality instance.
    by_personality: dict = {}
    for ship in snapshot:
        if not (ship.is_alive() and ship.brain
                and isinstance(ship.brain.personality, MLPersonality)
                and ship.brain._tick_timer - dt <= 0.0):
            continue
        pid = id(ship.brain.personality)
        if pid not in by_personality:
            by_personality[pid] = (ship.brain.personality, [])
        by_personality[pid][1].append(ship)

    for personality, ships in by_personality.values():
        X = np.stack([state_to_vector(build_ship_state(s, registry)) for s in ships])
        weights = personality.forward_batch(X)  # (N, 4)
        for ship, row in zip(ships, weights):
            ship.brain.overrides = dict(zip(personality.MODULE_NAMES, row.tolist()))
            ship.brain._batch_ticked = True


# ---------------------------------------------------------------------------
# Headless episode runner
# ---------------------------------------------------------------------------

def run_scenario(config: ScenarioConfig) -> dict:
    """
    Run one full headless episode from *config*.

    Returns a results dict:
    {
        "scenario":   str,
        "timestamp":  str   (ISO-8601),
        "outcome": {
            "winner":   int | null,   # faction id, null = draw / timeout
            "draw":     bool,
            "steps":    int,
            "simulated_seconds": float,
        },
        "factions": {
            "<id>": {"initial": int, "remaining": int}
        },
        "timeline": [                 # empty if record_interval == 0
            {"step": int, "factions": {"<id>": int}},
            ...
        ]
    }
    """
    sim   = config.simulation
    out   = sim.output
    state = setup_scenario(config)
    initial_counts = state['initial_counts']

    dt       = 1.0 / 60.0
    step     = 0
    timeline = []

    def _snapshot():
        return {str(fid): len(ShipRegistry.ships_by_faction.get(fid, []))
                for fid in initial_counts}

    if out.record_interval > 0:
        timeline.append({"step": 0, "factions": _snapshot()})

    while True:
        # --- Termination checks ---
        alive_factions = [
            fid for fid in initial_counts
            if ShipRegistry.ships_by_faction.get(fid)
        ]
        if len(alive_factions) <= 1:
            break
        if sim.max_steps is not None and step >= sim.max_steps:
            break

        # --- Update ---
        snapshot = list(ShipRegistry.ships)
        ShipRegistry.rebuild_ships_grid()
        _batch_ml_tick(snapshot, ShipRegistry, dt)
        for ship in snapshot:
            if ship.is_alive() and ship.brain:
                ship.brain.update(ship, dt, ShipRegistry)
        for ship in snapshot:
            if ship.is_alive() and ship.weapon:
                ship.weapon.update(ship, dt, ShipRegistry)
        for ship in ShipRegistry.ships:
            ship.update_target(ShipRegistry)
        for ship in snapshot:
            if ship.is_alive():
                ship.update(dt)

        step += 1

        # --- Timeline snapshot ---
        if out.record_interval > 0 and step % out.record_interval == 0:
            timeline.append({"step": step, "factions": _snapshot()})

    # --- Build result ---
    final_counts = {
        fid: len(ShipRegistry.ships_by_faction.get(fid, []))
        for fid in initial_counts
    }
    alive_factions = [fid for fid, n in final_counts.items() if n > 0]

    if len(alive_factions) == 1:
        winner, draw = alive_factions[0], False
    else:
        winner, draw = None, True

    result = {
        "scenario":  config.name,
        "timestamp": datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        "outcome": {
            "winner":            winner,
            "draw":              draw,
            "steps":             step,
            "simulated_seconds": round(step * dt, 2),
        },
        "factions": {
            str(fid): {"initial": initial_counts[fid], "remaining": final_counts[fid]}
            for fid in initial_counts
        },
        "timeline": timeline,
    }

    # --- Write output file if configured ---
    if out.file:
        os.makedirs(os.path.dirname(out.file) if os.path.dirname(out.file) else '.', exist_ok=True)
        with open(out.file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Results written to {out.file}")

    return result
