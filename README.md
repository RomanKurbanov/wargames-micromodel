# Wargames Micro-Model

A pipeline for distilling ChatGPT's tactical decision-making into a fast neural network. A 2D autonomous fleet battle sandbox (`wargames_2d/`) generates synthetic training data, ChatGPT labels it with steering weights, and a small MLP is trained to replicate those decisions at runtime.

## Overview

Ships from multiple factions fight autonomously using a modular steering system. Each ship's brain computes a weighted sum of behavior vectors every frame. A **personality** layer intercepts this process and overrides module weights based on a frozen snapshot of the ship's observable world (`ShipState`). The micro-model replaces hand-written personality rules with a learned approximation of ChatGPT's tactical reasoning.

```
Simulation → ShipState snapshots → ChatGPT labels → MLP training → MLPersonality in-game
```

## Setup

The scenario generation and ChatGPT labeling steps require an OpenAI API key. Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

The pipeline scripts load this file automatically at startup.

## Quick Start

```bash
# Simulation only
cd wargames_2d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py

# Full ML pipeline (from project root)
pip install -r requirements-generator.txt   # scenario generation
pip install -r requirements-training.txt    # model training

python generate_scenarios.py                # 1. generate battle scenarios via ChatGPT
python collect_data.py \
  --scenarios wargames_2d/scenarios/generated/ \
  --personality tactical2                   # 2. collect unlabeled ship-state data
python label_with_chatgpt.py \
  --input data/unlabeled/ \
  --output data/labeled/ \
  --mode stratified                         # 3. label with ChatGPT reasoning API
python analyze_labels.py                    # 4. inspect label quality
python train_model.py \
  --hidden 64 --epochs 1000 \
  --output models/ship_brain_v5.npz         # 5. train and export model
```

## Simulation Controls

| Input | Action |
|---|---|
| WASD | Pan camera |
| Scroll wheel | Zoom |
| Left-click ship | Select — shows weight overlay, state overlay, module vectors, range ring |
| Space | Pause / resume |
| ESC | Quit |

## Directory Structure

```
wargames-micromodel/
├── wargames_2d/                # Pygame simulation (self-contained)
│   ├── main.py                 # Pygame loop, rendering, HUD overlays
│   ├── simulation.py           # Headless runner (no display dependency)
│   ├── spawner.py              # Faction and brain setup
│   ├── registry.py             # Global entity store (ShipRegistry)
│   ├── scenario.py             # JSON → ScenarioConfig dataclass
│   ├── objective.py            # Shared team goal (area or target ship)
│   ├── obstacle_spawner.py     # Poisson-disk asteroid placement
│   ├── entities/
│   │   ├── ship.py             # Stats, combat, target acquisition
│   │   ├── weapon.py           # Cassette fire model, LoS, tracers
│   │   └── obstacle.py         # Static world geometry
│   ├── behavior/
│   │   ├── ship_brain.py       # Weight aggregator + _aviate() flight model
│   │   ├── steering_module.py  # SteeringModule ABC
│   │   ├── personality/        # ShipState, Personality ABC, ML/rule personalities
│   │   ├── navigate/           # ObjectiveModule, BoundsModule, CollisionAvoidance, Separation
│   │   ├── flock/              # AlignmentModule, CohesionModule
│   │   └── engage/             # EvasionModule, FiringSolutionModule
│   └── scenarios/              # JSON scenario files
│       ├── default.json        # 30v30 symmetric battle
│       ├── asymmetric.json     # Fast swarm vs slow snipers
│       └── generated/          # ChatGPT-authored tactical scenarios
├── generate_scenarios.py       # ChatGPT scenario authoring
├── collect_data.py             # Headless data capture (states.jsonl / states.npz)
├── label_with_chatgpt.py       # ChatGPT reasoning-API labeler
├── analyze_labels.py           # Label quality vs TacticalPersonality2
├── train_model.py              # MLX training → .npz export
├── models/                     # Trained weight files (.npz)
│   └── ship_brain_v4.npz       # Current production model
└── data/
    ├── unlabeled/              # Raw state snapshots
    ├── labeled/                # ChatGPT-labeled (labels.jsonl)
    └── labeled_t2/             # TacticalPersonality2-labeled
```

## Steering Pipeline

Each frame per ship:

1. Personality produces `{ClassName: float}` weight overrides from a frozen `ShipState` snapshot.
2. Effective weight per module = override if present, else `module.weight` (static default).
3. All `compute_steering(ship, registry) → Vector2` results are scaled and **summed unsimplified**.
4. The summed vector feeds `_aviate()`: direction = desired heading, magnitude (clamped to 1.0) = throttle.

Module static weight defaults:

| Module | Default weight | Role |
|---|---|---|
| `ObjectiveModule` | 0.8 | Pull toward team objective |
| `BoundsModule` | 1.0 | Hard world-boundary repulsion (up to 2.0) |
| `CollisionAvoidanceModule` | 0.9 | Avoid obstacles and ships |
| `SeparationModule` | 0.5 | Maintain minimum spacing from allies |
| `AlignmentModule` | 0.3 | Match heading of nearby allies |
| `CohesionModule` | 0.15 | Stay grouped with allies |
| `EvasionModule` | 0.6 | Flee enemies within weapon range |
| `FiringSolutionModule` | 0.8 | Close to firing position on target |

## Personality System

`Personality` is the sole extension point. It receives an immutable `ShipState` and returns weight overrides — it never touches live game objects.

```python
class MyPersonality(Personality):
    def get_weights(self, state: ShipState) -> dict[str, float]:
        if not state.weapon_loaded:
            return {'EvasionModule': 0.9, 'AttackModule': 0.0}
        return {'AttackModule': 1.0, 'EvasionModule': 0.2}
```

Available personalities:

| Identifier | Class | Description |
|---|---|---|
| `"default"` | `DefaultPersonality` | Retreat/suppress when reloading; push attack when loaded |
| `"none"` | — | Static module defaults only |
| `"tactical2"` | `TacticalPersonality2` | Rule-based teacher/oracle used during data collection |
| `"ml:<path>"` | `MLPersonality` | Runs trained MLP from `.npz` weights file |

Hot-swap mid-battle: `ship.brain.personality = new_personality`.

## ShipState Input Vector

`state_to_vector(state)` produces a 20-float `float32` array:

| Index | Field | Range |
|---|---|---|
| 0 | `hp_frac` | [0, 1] |
| 1 | `shield_frac` | [0, 1] |
| 2 | `weapon_loaded` | {0, 1} |
| 3 | `ammo_frac` | [0, 1] |
| 4 | `reload_progress` | [0, 1] |
| 5 | `speed_frac` | [0, 1] |
| 6 | `has_target` | {0, 1} |
| 7 | `norm_dist(target_distance)` | [0, 1] |
| 8 | `target_hp_frac` | [0, 1] |
| 9 | `tanh(closing_speed / 100)` | (-1, 1) |
| 10 | `can_fire` | {0, 1} |
| 11 | `friend_count / 30` | [0, 1] |
| 12 | `enemy_count / 30` (visual range only) | [0, 1] |
| 13 | `norm_dist(objective_distance)` | [0, 1] |
| 14 | `inside_objective` | {0, 1} |
| 15 | `max_hp / 265` | [0, 1] |
| 16 | `max_speed / 130` | [0, 1] |
| 17 | `weapon_range / 610` | [0, 1] |
| 18 | `weapon_damage / 38` | [0, 1] |
| 19 | `reload_duration / 11` | [0, 1] |

## Model Architecture

```
Linear(20, 64) → Tanh → Linear(64, 4) → Sigmoid
```

1,604 parameters. Outputs weights for 4 tactical modules: `ObjectiveModule`, `FlockingModule`, `EvasionModule`, `AttackModule`. `BoundsModule` and `CollisionAvoidanceModule` are static engine weights excluded from model output.

Current production: `models/ship_brain_v4.npz` — hidden=64, trained on ~1,400 ChatGPT reasoning-labeled rows.

## Scenario Format

Scenarios are JSON files that fully specify a battle. The `"personality"` field per faction accepts `"default"`, `"none"`, or `"ml:path/to/model.npz"` (path relative to `wargames_2d/`).

```json
{
  "name": "Example",
  "world_bounds": 2000,
  "factions": [
    {
      "id": 0,
      "ship_count": 20,
      "spawn_center": [500, 1000],
      "personality": "default",
      "ship": { "max_speed": 120, "hp": 80 },
      "weapon": { "range": 350, "damage": 12 }
    }
  ],
  "obstacles": { "enabled": true, "count": 40, "seed": 42 },
  "simulation": { "max_steps": 18000 }
}
```

See `wargames_2d/SCENARIOS.md` for the full schema.

## Data Pipeline Details

- **State capture rate**: ~4 Hz per ship (every ~15 frames at 60 fps)
- **Labeling API**: OpenAI `client.responses.create` with `reasoning={"effort": "medium"}` — avoids the 200k TPM ceiling of `chat.completions`
- **Label format** (`labels.jsonl`): one JSON object per row with `state_vector` (20 floats) and `weights` (4 floats for tactical modules)
- **`collect_data.py` note**: `ml:` personalities in scenario JSON are auto-replaced with `"tactical2"` before setup — paths would be invalid relative to the project root

## Further Reading

- `wargames_2d/DEVELOPER.md` — steering module authoring, weapon LoS, entity extension, debug techniques
- `wargames_2d/ML_INTEGRATION.md` — full input/output spec, model architecture, training approaches, headless runner example
- `wargames_2d/SCENARIOS.md` — complete JSON schema, result format, batch runner examples
