# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project develops a pipeline to generate synthetic training data for a micro-model that distills ChatGPT's decision-making into a fast neural network. The `wargames_2d/` directory is a testbed for this pipeline: it simulates autonomous ship battles, captures ship state snapshots during runtime, and generates synthetic data that describes what each ship sees (its parameters, nearby ships, objectives). This data is sent to ChatGPT with a prompt asking it to reason about the ship's course of action and output high-level steering weights. The ChatGPT responses are collected as training data, and a small ML model is trained to predict ChatGPT's outputs from raw ship states. This trained model is then plugged into `wargames_2d` as a "ship personality" to replace static baseline weights.

## Commands

```bash
# Setup (in wargames_2d directory)
cd wargames_2d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run interactive simulation
python main.py

# Run headless simulation with scenario
python main.py --headless --scenario scenarios/default.json --output results.json

# Run headless batch episodes
python simulation.py
```

Runtime controls in the window: WASD to pan camera · left-click a ship to inspect weights · ESC to quit.

## Architecture

### Core Simulation Files (wargames_2d/)

```
main.py                       # PyGame loop, rendering, camera, ship selection, weight overlay
registry.py                   # Global object/ship lists (ShipRegistry class-level store)
objective.py                  # Shared team goal (area or target-ship)
spawner.py                    # Faction setup: ships, weapons, brains
obstacle_spawner.py           # Poisson-disk asteroid placement with guaranteed spacing

entities/
  object.py                   # Base entity: position, size, auto-register to ShipRegistry
  ship.py                     # Ship stats, combat, target acquisition, destroy()
  weapon.py                   # Cassette fire model, LoS check, tracers
  obstacle.py                 # Static world geometry

behavior/
  steering_module.py          # SteeringModule ABC + safe_normalize()
  ship_brain.py               # ShipBrain: weight aggregator + _aviate() flight model
  personality.py              # ShipState frozen dataclass + Personality ABC + DefaultPersonality
  navigate/                   # ObjectiveModule, BoundsModule, CollisionAvoidanceModule, SeparationModule
  flock/                      # AlignmentModule, CohesionModule (friendly neighbors)
  engage/                     # EvasionModule, FiringSolutionModule (enemy neighbors)

factions.py                   # faction int → RGB color mapping
scenario.py                   # JSON parsing into ScenarioConfig dataclass tree
simulation.py                 # setup_scenario() and run_scenario() for headless execution
```

### Steering Pipeline (Core AI)

Each frame per ship (`ShipBrain.update()` in `behavior/ship_brain.py`):

1. `compute_weight_overrides(ship, registry)` → calls `personality.get_weights()` if present, returns `{ClassName: float}` overrides
2. For each `SteeringModule`, effective weight = override if present, else `module.weight`
3. Weights below 0 or non-finite are silently dropped
4. All `module.compute_steering(ship, registry)` results are scaled by their weight and **summed unsimplified** into one `Vector2`
5. The summed vector feeds `_aviate()` which handles rotation, velocity decomposition, lateral damping, and position integration

Importantly: `BoundsModule` can return magnitude up to 2.0 (scales by `t * 2.0`), which can overpower other modules at the world boundary.

### Personality System

`Personality` is the sole extension point for AI behavior. It receives a frozen `ShipState` snapshot and returns weight overrides.

```
Live objects → build_ship_state() → ShipState (frozen dataclass) → personality.get_weights() → {ClassName: float}
```

`ShipState` has 15 pre-normalized fields: hp_frac, shield_frac, weapon_loaded, ammo_frac, reload_progress, can_fire, speed_frac, has_target, target_distance, target_hp_frac, closing_speed, friend_count, enemy_count, objective_distance, inside_objective.

`DefaultPersonality` (built-in): boosts evasion to 0.9 and suppresses combat/objective/flock when reloading; restores firing solution to 1.0 / evasion to 0.3 when loaded.

Hot-swapping mid-battle is safe: `ship.brain.personality = new_personality`.

### Entity Registration

`Object.__init__` auto-registers in `ShipRegistry.objects`. `Ship.__init__` also registers in `ships` and `ships_by_faction[faction]`. `Ship.destroy()` unregisters from both. `ShipRegistry.clear()` wipes all three — called by `setup_scenario()` between headless episodes.

Any new entity that should block movement and shots only needs to extend `Object`.

### Update Order (per frame)

1. `ShipBrain.update()` — steering → `_aviate()` → new position/velocity
2. `Weapon.update()` — fire/reload state machine, tracer TTL decay
3. `Ship.update_target()` — nearest enemy scan within `visual_range`

Modules must not write to `ship` or `registry` during `compute_steering` — that would break this order.

### Scenario System

Scenarios are JSON files parsed into a `ScenarioConfig` dataclass tree (`scenario.py`). `simulation.py` provides `setup_scenario()` (shared init, calls `ShipRegistry.clear()`) and `run_scenario()` (headless loop at fixed `dt = 1/60`, returns results dict).

Each faction's `"personality"` field accepts: `"default"` (built-in rule-based), `"none"` (static weights only), or `"ml:<path/to/model.npz>"` (loads `MLPersonality`). The `ml:` spec is wired in `simulation._resolve_personality()` but `behavior/personality/ml_personality.py` does not exist yet — the reference implementation is in `wargames_2d/ML_INTEGRATION.md`.

## ML Pipeline (Micro-Model Distillation)

The project's core purpose is to distill ChatGPT's reasoning into a fast micro-model:

1. **Scenario Generation**: Develop a scenario generator (likely with ChatGPT help) that creates diverse wargames scenarios
2. **Simulation & Data Capture**: Run simulations in `wargames_2d`, capturing ship state snapshots during runtime at intervals
3. **Synthetic Data Creation**: Format snapshots to describe what each ship sees (parameters, nearby ships, objectives)
4. **ChatGPT Reasoning**: Send ship state data + prompt to ChatGPT asking it to reason about the ship's course of action
5. **Output Collection**: ChatGPT outputs high-level steering weights (module weights in `[0, 1]`)
6. **Model Training**: Train a small ML model on the raw inputs → ChatGPT outputs pairs
7. **Deployment**: Plug the trained model into `wargames_2d` as a `Personality` to replace static baseline weights

The `models/` and `training/` directories don't exist yet — create them when starting ML work. See `wargames_2d/ML_INTEGRATION.md` for the full input normalization spec (`state_to_vector`), recommended architecture (15→24→8, ~392 params), and training approaches (supervised imitation, neuroevolution, PPO).

### Steering Pipeline Integration

The steering pipeline sums module outputs unsimplified into a single vector. The direction determines desired heading; magnitude (clamped to 1.0) controls throttle. The trained micro-model will output the same weight format that `DefaultPersonality` uses — a dict mapping module class names to floats.

### Important Conventions

- **No `NeutralBaseModule`**: `CollisionAvoidanceModule` and `SeparationModule` extend `SteeringModule` directly
- **`CollisionAvoidanceModule` ignores its `neighbor_radius` param**: The cutoff is `soft_r = (ship.size + obj.size) * 3.0`, computed per-object from sizes
- **`ship.objective` duality**: Can be an `Objective` object or a legacy raw `Vector2`. `ObjectiveModule` checks `hasattr(ship.objective, 'position')` to distinguish them
- **Adding a steering module**: Implement `SteeringModule`, place under `behavior/`, register in `spawner._make_brain()`, add display name in `main.py`'s `_MODULE_LABELS` dict
- **`AlignmentModule` filters reloading ships**: Only aligns with neighbors whose weapon is not reloading
- **Module unit testing without a display**: `pygame.init()` is sufficient to use `pygame.math.Vector2` without opening a window. Stub `_FakeShip` and `_FakeRegistry` objects are enough to test any module in isolation
- **Obstacle layout reproducibility**: `obstacle_spawner.spawn_obstacles(seed=42)` freezes the layout for iterating on steering behavior

## File Ownership Quick Reference

| Want to change… | Edit… |
|---|---|
| Ship movement feel | `entities/ship.py` stats, `behavior/ship_brain._aviate` |
| Weapon fire model | `entities/weapon.py` |
| AI decision logic | `behavior/ship_brain.compute_weight_overrides` |
| Individual behaviors | files under `behavior/navigate/`, `flock/`, `engage/` |
| World layout | `main.py` spawn centers, `obstacle_spawner.py` params |
| Faction colors | `factions.py` |
| Rendering | `main.py` render section |
| Entity registration | `registry.py`, `entities/object.py` |
| Scenario generation | New scripts in project root for generating diverse scenarios |
| Data capture pipeline | New scripts to snapshot ship states and format for ChatGPT |
| ML training pipeline | New scripts for training the micro-model on ChatGPT outputs |

## Detailed Documentation (wargames_2d/)

- `DEVELOPER.md` — steering module authoring, personality examples, weapon LoS, entity extension, debug techniques
- `ML_INTEGRATION.md` — input normalization, output spec, model architecture, training approaches, headless episode runner
- `SCENARIOS.md` — full JSON schema for scenario files, result format, batch runner examples
