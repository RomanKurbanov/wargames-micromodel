"""
Personality — external weight policy for ShipBrain.

A Personality receives an immutable ShipState snapshot each frame and
returns per-module weight overrides {ClassName: float}.  It never
receives a reference to any live game object, so it cannot modify game
state or steer vectors directly.

Extension point:
    Subclass Personality and implement get_weights().
    Pass the instance to ShipBrain(personality=...) at spawn time.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import pygame


# ---------------------------------------------------------------------------
# Snapshot — the only thing a personality is allowed to see
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ShipState:
    """
    Immutable per-frame snapshot of one ship's observable world.
    All values are pre-computed and normalized to [0, 1] or natural units
    so personalities don't need to know internal game constants.

    Distances are in world pixels.  Speeds are in pixels/second.
    """

    # --- Health ---
    hp_frac: float          # current HP / max HP              [0–1]
    shield_frac: float      # current shield / max shield      [0–1]

    # --- Weapon ---
    weapon_loaded: bool     # True = cassette ready to fire
    ammo_frac: float        # shots remaining / cassette_size  [0–1]
    reload_progress: float  # 0 = reload just started, 1 = full [0–1]

    # --- Motion ---
    speed_frac: float       # current speed / max_speed        [0–1]

    # --- Target ---
    has_target: bool
    target_distance: float  # pixels; math.inf when no target
    target_hp_frac: float   # target HP fraction               [0–1]; 1.0 when no target
    closing_speed: float    # px/s, positive = closing in;     0.0 when no target
    can_fire: bool          # weapon is in range / arc / LoS right now

    # --- Team (fleet-wide, not radius-limited) ---
    friend_count: int       # alive friendlies excluding self
    enemy_count: int        # alive enemies across all factions

    # --- Objective ---
    objective_distance: float   # px to objective centre; math.inf if none
    inside_objective: bool      # ship is inside the patrol ring


# ---------------------------------------------------------------------------
# Snapshot builder — extracts data from live objects, returns frozen copy
# ---------------------------------------------------------------------------

def build_ship_state(ship, registry) -> ShipState:
    """
    Construct a ShipState from live game objects.
    Called by ShipBrain before invoking the personality.
    Not part of the public personality API.
    """

    # Health (normalised against per-ship max values)
    hp_frac     = max(0.0, min(1.0, ship.hp    / 100.0))
    shield_frac = (max(0.0, min(1.0, ship.shield / ship.max_shield))
                   if ship.max_shield > 0 else 0.0)

    # Weapon
    w = ship.weapon
    if w is not None:
        weapon_loaded    = not w.reloading
        ammo_frac        = (w.cassette_size - w.shots_fired) / max(w.cassette_size, 1)
        reload_progress  = (
            1.0 - w.reload_timer / max(w.reload_duration, 1e-6)
            if w.reloading else 1.0
        )
        can_fire = (
            w.can_fire_at(ship, ship.target.position, registry)
            if (ship.target is not None and ship.target.is_alive() and not w.reloading)
            else False
        )
    else:
        weapon_loaded   = True
        ammo_frac       = 1.0
        reload_progress = 1.0
        can_fire        = False

    # Motion
    speed_frac = ship.velocity.length() / max(ship.max_speed, 1e-6)

    # Target
    if ship.target is not None and ship.target.is_alive():
        to_target       = ship.target.position - ship.position
        target_distance = to_target.length()
        target_hp_frac  = max(0.0, min(1.0, ship.target.hp / 100.0))
        closing_speed   = (
            (ship.velocity - ship.target.velocity).dot(to_target / target_distance)
            if target_distance > 1e-4 else 0.0
        )
        has_target = True
    else:
        target_distance = math.inf
        target_hp_frac  = 1.0
        closing_speed   = 0.0
        has_target      = False

    # Team counts
    friend_count = sum(
        1 for s in registry.ships_by_faction.get(ship.faction, [])
        if s is not ship
    )
    enemy_count = sum(
        len(lst)
        for f, lst in registry.ships_by_faction.items()
        if f != ship.faction
    )

    # Objective
    obj = ship.objective
    if obj is not None and obj.is_valid():
        obj_dist            = (obj.position - ship.position).length()
        objective_distance  = obj_dist
        inside_objective    = obj_dist <= obj.patrol_radius
    else:
        objective_distance = math.inf
        inside_objective   = False

    return ShipState(
        hp_frac=hp_frac,
        shield_frac=shield_frac,
        weapon_loaded=weapon_loaded,
        ammo_frac=ammo_frac,
        reload_progress=reload_progress,
        speed_frac=speed_frac,
        has_target=has_target,
        target_distance=target_distance,
        target_hp_frac=target_hp_frac,
        closing_speed=closing_speed,
        can_fire=can_fire,
        friend_count=friend_count,
        enemy_count=enemy_count,
        objective_distance=objective_distance,
        inside_objective=inside_objective,
    )


# ---------------------------------------------------------------------------
# Personality — the public ABC
# ---------------------------------------------------------------------------

class Personality(ABC):
    """
    Override this class to give a faction (or individual ship) a personality.

    get_weights() is the only method.  It receives a ShipState snapshot —
    an immutable value object — and must return a plain dict mapping
    module class names to floats.

    Rules:
    - Return values should be in [0.0, 1.0].  The brain floors negatives
      to 0 and drops NaN/Inf, but clean values make intent explicit.
    - Missing keys leave that module's static default weight unchanged.
    - Do not store mutable references to game objects in the personality;
      everything you need is in the ShipState.
    """

    @abstractmethod
    def get_weights(self, state: ShipState) -> dict[str, float]:
        ...


# ---------------------------------------------------------------------------
# Built-in personality — ammo-state rotation (previously hard-coded in brain)
# ---------------------------------------------------------------------------

class DefaultPersonality(Personality):
    """
    Reloading ships retreat and go quiet.
    Loaded ships push for a firing solution.
    """

    def get_weights(self, state: ShipState) -> dict[str, float]:
        if not state.weapon_loaded:
            return {
                'EvasionModule':        0.9,
                'FiringSolutionModule': 0.0,
                'ObjectiveModule':      0.0,
                'AlignmentModule':      0.05,
                'CohesionModule':       0.05,
            }
        return {
            'FiringSolutionModule': 1.0,
            'EvasionModule':        0.3,
        }
