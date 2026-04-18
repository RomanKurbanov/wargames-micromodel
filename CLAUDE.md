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

`ShipState` has 15 per-ship fields. `state_to_vector()` in `behavior/personality/__init__.py` expands this to **20 floats** by appending 5 normalized faction stats: max_hp/265, max_speed/130, weapon_range/610, weapon_damage/38, reload_duration/11. All ML scripts use the 20-float vector.

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

Each faction's `"personality"` field accepts: `"default"` (built-in rule-based), `"none"` (static weights only), or `"ml:<path/to/model.npz>"` (loads `MLPersonality`). The `ml:` spec is wired in `simulation._resolve_personality()`. `behavior/personality/ml_personality.py` is implemented.

**IMPORTANT**: `ml:` paths in scenario JSON are relative to `wargames_2d/`. When running `collect_data.py` from the project root, these paths fail. `collect_data.py` auto-replaces `ml:` personalities with `"tactical2"` before `setup_scenario()` — do not remove this pre-processing step.

## ML Pipeline (Micro-Model Distillation)

The ML pipeline is operational. Key scripts (all run from project root):

```bash
# 1. Generate scenarios
python generate_scenarios.py

# 2. Collect labeled data (TacticalPersonality2 as teacher → labels.jsonl)
python collect_data.py --scenarios wargames_2d/scenarios/generated/ --personality tactical2

# 3. Label with ChatGPT reasoning API (stratified sampling, --append to top up)
python label_with_chatgpt.py --input data/unlabeled/ --output data/labeled/ --mode stratified
python label_with_chatgpt.py --input data/unlabeled/ --output data/labeled/ --mode stratified --append

# 4. Analyze label quality vs TacticalPersonality2
python analyze_labels.py

# 5. Train model (MLX on Apple Silicon, exports .npz for pure-numpy inference)
python train_model.py --hidden 64 --epochs 1000 --output models/ship_brain_v5.npz
```

Data layout: `data/labeled/` (ChatGPT-labeled, `labels.jsonl`), `data/labeled_t2/` (TacticalPersonality2-labeled), `data/unlabeled/` (raw `states.jsonl` + `states.npz`).

Labeling uses `client.responses.create` (OpenAI Responses API) with `reasoning={"effort": "medium"}` — NOT `chat.completions`. The responses API avoids the 200k TPM ceiling that caused ~70% failures with chat.completions.

The `models/` directory contains trained `.npz` weight files. Current production model: `models/ship_brain_v4.npz` (hidden=64, trained on ~1400 reasoning-labeled rows). Architecture: **Linear(20,64) → Tanh → Linear(64,4) → Sigmoid**, 1604 params. See `wargames_2d/ML_INTEGRATION.md` for background. NOTE: `ship_brain_v2.npz` has incompatible shape (6 outputs, hidden=128) — do not use with MLPersonality.

### Steering Pipeline Integration

The steering pipeline sums module outputs unsimplified into a single vector. The direction determines desired heading; magnitude (clamped to 1.0) controls throttle. The trained micro-model will output the same weight format that `DefaultPersonality` uses — a dict mapping module class names to floats.

### Important Conventions

- **`visual_range` detection circle**: When a ship is selected in `main.py`, a green ring at `ship.visual_range * zoom` px is drawn — correctly scales with camera zoom.
- **`LABEL_MODULE_NAMES`**: Only 4 tactical modules (Objective, Flocking, Evasion, Attack) are written to `labels.jsonl`. BoundsModule and CollisionAvoidanceModule are static engine weights, excluded from model outputs.
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
| Scenario generation | `generate_scenarios.py` |
| Data capture pipeline | `collect_data.py` |
| ChatGPT labeling | `label_with_chatgpt.py` |
| Label quality analysis | `analyze_labels.py` |
| ML training pipeline | `train_model.py` |
| ML inference personality | `wargames_2d/behavior/personality/ml_personality.py` |
| Teacher/oracle personality | `wargames_2d/behavior/personality/tactical_personality2.py` |

## Detailed Documentation (wargames_2d/)

- `DEVELOPER.md` — steering module authoring, personality examples, weapon LoS, entity extension, debug techniques
- `ML_INTEGRATION.md` — input normalization, output spec, model architecture, training approaches, headless episode runner
- `SCENARIOS.md` — full JSON schema for scenario files, result format, batch runner examples
