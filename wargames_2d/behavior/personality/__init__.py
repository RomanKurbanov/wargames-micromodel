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

    # --- Team (within visual_range only) ---
    friend_count: int       # alive friendlies within visual range, excluding self
    enemy_count: int        # alive enemies within visual range

    # --- Objective ---
    objective_distance: float   # px to objective centre; math.inf if none
    inside_objective: bool      # ship is inside the patrol ring

    # --- Ship profile (per-ship absolute stats; 0 when field is absent/inapplicable) ---
    max_hp:           float   # ship's maximum HP                       (HP)
    max_speed_stat:   float   # ship's max speed                        (px/s)
    weapon_range:     float   # weapon range                            (px)
    weapon_damage:    float   # weapon damage per shot                  (dmg)
    reload_duration:  float   # weapon reload time                      (s)


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
    hp_frac     = max(0.0, min(1.0, ship.hp / max(ship.max_hp, 1e-6)))
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
        weapon_range_val    = w.range
        weapon_damage_val   = w.damage
        reload_duration_val = w.reload_duration
    else:
        weapon_loaded       = True
        ammo_frac           = 1.0
        reload_progress     = 1.0
        can_fire            = False
        weapon_range_val    = 0.0
        weapon_damage_val   = 0.0
        reload_duration_val = 0.0

    # Motion
    speed_frac = ship.velocity.length() / max(ship.max_speed, 1e-6)

    # Target
    if ship.target is not None and ship.target.is_alive():
        to_target       = ship.target.position - ship.position
        target_distance = to_target.length()
        target_hp_frac  = max(0.0, min(1.0, ship.target.hp / max(ship.target.max_hp, 1e-6)))
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

    # Team counts — only ships within visual range
    vr = ship.visual_range
    friend_count = sum(
        1 for s in registry.ships_by_faction.get(ship.faction, [])
        if s is not ship and (s.position - ship.position).length() <= vr
    )
    enemy_count = sum(
        1 for f, lst in registry.ships_by_faction.items()
        if f != ship.faction
        for s in lst
        if (s.position - ship.position).length() <= vr
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
        max_hp=ship.max_hp,
        max_speed_stat=ship.max_speed,
        weapon_range=weapon_range_val,
        weapon_damage=weapon_damage_val,
        reload_duration=reload_duration_val,
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
                'EvasionModule':   0.9,
                'AttackModule':    0.0,
                'ObjectiveModule': 0.0,
                'FlockingModule':  0.05,
            }
        return {
            'AttackModule':    1.0,
            'EvasionModule':   0.3,
        }


# ---------------------------------------------------------------------------
# ML utilities — shared by LoggingPersonality and MLPersonality
# ---------------------------------------------------------------------------

import numpy as np  # noqa: E402 — intentionally after game classes

MODULE_NAMES = [
    'ObjectiveModule',
    'BoundsModule',
    'CollisionAvoidanceModule',
    'FlockingModule',
    'EvasionModule',
    'AttackModule',
]

_DIST_SCALE        = 400.0   # typical weapon range — normalises target/objective distances
_SPEED_SCALE       = 100.0   # typical max speed    — normalises closing_speed via tanh
_COUNT_SCALE       = 30.0    # ships per faction at full strength
_MAX_HP_REF        = 265.0   # maximum possible HP
_MAX_SPEED_REF     = 130.0   # maximum possible max_speed (px/s)
_WEAPON_RANGE_REF  = 610.0   # maximum possible weapon range (px)
_WEAPON_DMG_REF    = 38.0    # maximum possible damage per shot
_RELOAD_DUR_REF    = 11.0    # maximum possible reload duration (s)


def state_to_vector(state: ShipState) -> np.ndarray:
    """
    Convert a ShipState to a 20-element float32 numpy vector.

    All values are in [0, 1] except closing_speed which is tanh-normalised
    to (-1, +1).  math.inf distances map to 0.0 (far away = low signal).

    Indices:
      0  hp_frac             7  norm_dist(target_distance)   14  inside_objective
      1  shield_frac         8  target_hp_frac               15  max_hp / 265
      2  weapon_loaded       9  tanh(closing_speed/100)      16  max_speed / 130
      3  ammo_frac          10  can_fire                     17  weapon_range / 610
      4  reload_progress    11  friend_count / 30            18  weapon_damage / 38
      5  speed_frac         12  enemy_count / 30 (visual)    19  reload_duration / 11
      6  has_target         13  norm_dist(objective_distance)
    """
    def norm_dist(d: float) -> float:
        return 1.0 / (1.0 + d / _DIST_SCALE) if math.isfinite(d) else 0.0

    return np.array([
        state.hp_frac,                                              # 0
        state.shield_frac,                                          # 1
        float(state.weapon_loaded),                                 # 2
        state.ammo_frac,                                            # 3
        state.reload_progress,                                      # 4
        state.speed_frac,                                           # 5
        float(state.has_target),                                    # 6
        norm_dist(state.target_distance),                           # 7
        state.target_hp_frac,                                       # 8
        math.tanh(state.closing_speed / _SPEED_SCALE),              # 9
        float(state.can_fire),                                      # 10
        min(state.friend_count / _COUNT_SCALE, 1.0),                # 11
        min(state.enemy_count  / _COUNT_SCALE, 1.0),                # 12
        norm_dist(state.objective_distance),                        # 13
        float(state.inside_objective),                              # 14
        min(state.max_hp          / _MAX_HP_REF,       1.0),        # 15
        min(state.max_speed_stat  / _MAX_SPEED_REF,    1.0),        # 16
        min(state.weapon_range    / _WEAPON_RANGE_REF, 1.0),        # 17
        min(state.weapon_damage   / _WEAPON_DMG_REF,   1.0),        # 18
        min(state.reload_duration / _RELOAD_DUR_REF,   1.0),        # 19
    ], dtype=np.float32)
