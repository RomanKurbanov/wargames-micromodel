# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 2D autonomous fleet battle sandbox in Python/PyGame (`pygame>=2.5`). Ships from multiple factions fight using a modular steering/AI system. No player unit — pure simulation. Lives inside a larger Unity project tree but is entirely independent of Unity.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run interactive window (default two-faction battle)
python main.py

# Run a specific scenario
python main.py --scenario scenarios/default.json

# Headless run (no display), write results JSON
python main.py --headless --scenario scenarios/asymmetric.json --output results.json

# Batch runner / episode loop
python simulation.py
```

Runtime controls in the window: WASD to pan · left-click a ship to inspect weights · ESC to quit.

## Architecture

### Steering pipeline

This is the core of the AI and spans `ship_brain.py`, `personality.py`, and all `behavior/` modules. Each frame per ship:

1. `ShipBrain.compute_weight_overrides(ship, registry)` delegates to `personality.get_weights(ShipState)` if a personality is set, returning `{ClassName: float}` overrides.
2. For each `SteeringModule`, effective weight = override if present, else `module.weight`. Weights below 0 or non-finite are silently dropped.
3. All `module.compute_steering(ship, registry)` results are scaled by their weight and **summed unsimplified** into one `Vector2`. Magnitude matters — the summed vector's direction is the desired heading, its magnitude (clamped to 1.0) controls throttle.
4. The summed vector feeds `_aviate()` in `ShipBrain`, which handles rotation, velocity decomposition, lateral damping, and position integration.

Importantly: `BoundsModule` can return magnitude up to 2.0 (it scales by `t * 2.0`), which can overpower all other modules at the world boundary.

### Personality system

`Personality` is an ABC that only produces weights — it never touches live game objects. The data flow is enforced structurally:

```
Live objects → build_ship_state() → ShipState (frozen dataclass) → personality.get_weights() → {ClassName: float}
```

`ShipState` has 15 pre-normalized fields (hp_frac, weapon_loaded, ammo_frac, reload_progress, can_fire, speed_frac, has_target, target_distance, target_hp_frac, closing_speed, friend_count, enemy_count, objective_distance, inside_objective). Return weights in `[0, 1]`; the brain guards against NaN/Inf/negative values.

`DefaultPersonality` (built-in): boosts evasion to 0.9 and suppresses combat/objective/flock when reloading; restores firing solution to 1.0 / evasion to 0.3 when loaded.

A single personality instance can be shared across an entire faction. Hot-swapping mid-battle is safe (`ship.brain.personality = new_personality`).

### Entity registration

`Object.__init__` auto-registers in `ShipRegistry.objects`. `Ship.__init__` also registers in `ships` and `ships_by_faction[faction]`. `Ship.destroy()` unregisters from both. `ShipRegistry.clear()` wipes all three — called by `setup_scenario()` between headless episodes.

The `ships` list is used for targeting/flock queries; `objects` (ships + obstacles) is used for collision avoidance and weapon LoS checks. Any new entity that should block movement and shots only needs to extend `Object`.

### Update order (per frame)

1. `ShipBrain.update()` — steering → `_aviate()` → new position/velocity
2. `Weapon.update()` — fire/reload state machine, tracer TTL decay
3. `Ship.update_target()` — nearest enemy scan within `visual_range`

Modules must not write to `ship` or `registry` during `compute_steering` — that would break this order. The one sanctioned side-effect slot is `ship.course_lock` (currently a stub `Vector2(0,0)`).

### Scenario system

Scenarios are JSON files parsed into a `ScenarioConfig` dataclass tree (`scenario.py`). `simulation.py` provides `setup_scenario()` (shared init, calls `ShipRegistry.clear()`) and `run_scenario()` (headless loop at fixed `dt = 1/60`, returns results dict). `main.py` reads `--scenario`, `--headless`, `--output` flags.

## Important Conventions

- **No `NeutralBaseModule`**: Unlike the Unity version, `CollisionAvoidanceModule` and `SeparationModule` extend `SteeringModule` directly.
- **`CollisionAvoidanceModule` ignores its `neighbor_radius` param**: The cutoff is `soft_r = (ship.size + obj.size) * 3.0`, computed per-object from sizes. The constructor param is a dead parameter.
- **`ship.objective` duality**: Can be an `Objective` object or a legacy raw `Vector2`. `ObjectiveModule` checks `hasattr(ship.objective, 'position')` to distinguish them.
- **Adding a steering module**: Implement `SteeringModule`, place under `behavior/`, register in `spawner._make_brain()`, add display name in `main.py`'s `_MODULE_LABELS` dict.
- **`AlignmentModule` filters reloading ships**: Only aligns with neighbors whose weapon is not reloading, to prevent retreating ships from dragging the formation.
- **`MLPersonality`**: Referenced in `ML_INTEGRATION.md` and supported by the `"ml:<path>"` scenario field, but `behavior/personality/ml_personality.py` does not exist yet — it is documented as a code snippet only.
- **Module unit testing without a display**: `pygame.init()` is sufficient to use `pygame.math.Vector2` without opening a window. Stub `_FakeShip` and `_FakeRegistry` objects are enough to test any module in isolation.
- **Obstacle layout reproducibility**: `obstacle_spawner.spawn_obstacles(seed=42)` freezes the layout for iterating on steering behavior.
