# ML Integration Guide

This document is written for ML engineers picking up the project cold. It covers the contract between the simulation and any learned model, the input/output spec, training options, and a reference implementation.

No changes to the simulation engine are required to plug in a trained model.

---

## The Integration Point

Each ship's AI runs through a `Personality` object. Its only job is:

```
ShipState  →  get_weights()  →  dict[str, float]
```

`ShipState` is an immutable snapshot of one ship's observable world, built from live game objects before your code is called. The output dict maps steering module class names to weight floats in `[0, 1]`. The simulation engine sums the steering vectors scaled by those weights — your model never touches vectors, physics, or game state directly.

The full ABC lives in `behavior/personality/__init__.py`. To deploy a model, subclass `Personality`, implement `get_weights`, and pass an instance to `spawn_faction(..., personality=instance)`.

---

## Input Specification

### Raw ShipState fields

```python
@dataclass(frozen=True)
class ShipState:
    # --- Tactical state ---
    hp_frac:            float   # current HP / max HP
    shield_frac:        float   # current shield / max shield
    weapon_loaded:      bool    # True = cassette ready
    ammo_frac:          float   # shots remaining / cassette_size
    reload_progress:    float   # 0 = reload just started, 1 = complete
    speed_frac:         float   # current speed / max_speed
    has_target:         bool
    target_distance:    float   # pixels; math.inf when no target
    target_hp_frac:     float   # target HP fraction; 1.0 when no target
    closing_speed:      float   # px/s, positive = closing; 0 when no target
    can_fire:           bool    # weapon loaded, aimed, in range, clear LoS
    friend_count:       int     # alive friendlies excluding self
    enemy_count:        int     # alive enemies
    objective_distance: float   # pixels to objective centre; math.inf if none
    inside_objective:   bool    # ship is inside the patrol ring
    # --- Ship profile (per-ship absolute stats for model context) ---
    max_hp:             float   # ship's maximum HP (HP)
    max_speed_stat:     float   # ship's max speed (px/s)
    weapon_range:       float   # weapon range (px; 0 if no weapon)
    weapon_damage:      float   # weapon damage per shot (dmg; 0 if no weapon)
    reload_duration:    float   # weapon reload time (s; 0 if no weapon)
```

### Normalization to network input vector (20 floats)

`target_distance` and `objective_distance` can be `math.inf`. Feed raw values into a network and you get `nan` weights. Normalize before the forward pass.

The authoritative implementation lives in `behavior/personality/__init__.py` as `state_to_vector()`. Summary:

```python
import math
import numpy as np

_DIST_SCALE  = 400.0   # typical weapon range — tunes sensitivity
_SPEED_SCALE = 100.0   # max ship speed
_COUNT_SCALE = 30.0    # ships per faction at full strength

_MAX_HP_REF        = 265.0   # maximum possible HP in game
_MAX_SPEED_REF     = 130.0   # maximum possible max_speed
_WEAPON_RANGE_REF  = 610.0   # maximum possible weapon range
_WEAPON_DMG_REF    = 38.0    # maximum possible damage/shot
_RELOAD_DUR_REF    = 11.0    # maximum possible reload duration

def state_to_vector(state: ShipState) -> np.ndarray:
    def norm_dist(d: float) -> float:
        return 1.0 / (1.0 + d / _DIST_SCALE) if math.isfinite(d) else 0.0

    return np.array([
        state.hp_frac,                                          # 0
        state.shield_frac,                                      # 1
        float(state.weapon_loaded),                             # 2
        state.ammo_frac,                                        # 3
        state.reload_progress,                                  # 4
        state.speed_frac,                                       # 5
        float(state.has_target),                                # 6
        norm_dist(state.target_distance),                       # 7  — 0=far, 1=adjacent
        state.target_hp_frac,                                   # 8
        math.tanh(state.closing_speed / _SPEED_SCALE),          # 9  — bounded ±1
        float(state.can_fire),                                  # 10
        min(state.friend_count / _COUNT_SCALE, 1.0),            # 11
        min(state.enemy_count  / _COUNT_SCALE, 1.0),            # 12
        norm_dist(state.objective_distance),                    # 13
        float(state.inside_objective),                          # 14
        min(state.max_hp          / _MAX_HP_REF,       1.0),   # 15
        min(state.max_speed_stat  / _MAX_SPEED_REF,    1.0),   # 16
        min(state.weapon_range    / _WEAPON_RANGE_REF, 1.0),   # 17
        min(state.weapon_damage   / _WEAPON_DMG_REF,   1.0),   # 18
        min(state.reload_duration / _RELOAD_DUR_REF,   1.0),   # 19
    ], dtype=np.float32)
```

Feature 7 (`target_distance`) maps to 1 when the ship is at point-blank range and approaches 0 as range increases, matching the intuition that "closer = more urgent".

Feature 9 (`closing_speed`) is tanh-bounded: negative means the target is opening the distance, positive means it is closing in.

Features 15–19 are per-ship absolute stats. They let the model distinguish a long-range sniper from a short-range brawler from a fast-reload skirmisher. Stats are frozen at spawn time and do not change during the episode.

---

## Output Specification

### Module names and their role

The output dict must use these exact class name strings as keys. Missing keys leave that module's default static weight unchanged.

| Key | Default weight | What it does |
|---|---|---|
| `ObjectiveModule` | 0.8 | Pull toward the team's area objective |
| `BoundsModule` | 1.0 | Keep inside world boundary (hard safety) |
| `CollisionAvoidanceModule` | 0.9 | Avoid obstacles and other ships |
| `FlockingModule` | 0.3 | Stay with and align to the friendly group |
| `EvasionModule` | 0.6 | Pull away from enemy weapon range |
| `AttackModule` | 0.8 | Close to firing range of current target |

### Output layer

6 neurons, sigmoid activation. Output range is (0, 1).

The engine floors negatives to 0 and drops `NaN`/`Inf`, but clean sigmoid output avoids relying on those guards.

### What suppression means

Setting a weight to `0.0` removes that module's influence entirely for this frame. The ship will still be steered by all other active modules. You do not need to output all 6 keys — returning an empty dict leaves all modules at their static defaults.

`BoundsModule` and `CollisionAvoidanceModule` are safety modules. Suppressing them causes ships to fly out of bounds or into obstacles. `BoundsModule` must be ≥ 0.5 at all times.

---

## Model Architecture

The decision surface is not highly non-linear. A single hidden layer is sufficient.

### Recommended starting point

```
Input  →  Linear(20, 32)  →  Tanh  →  Linear(32, 6)  →  Sigmoid
```

**Parameter count: 870.** Inference time is negligible — a batched forward pass for all 60 ships takes under 0.1 ms on CPU with numpy.

Larger models (two hidden layers, 64 units) are unlikely to improve performance given the input dimensionality. If you find yourself needing more capacity, the bottleneck is probably the reward signal or training data quality, not the model.

### Pure numpy inference (no framework dependency)

```python
def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def forward(x: np.ndarray, W1, b1, W2, b2) -> np.ndarray:
    h = np.tanh(W1 @ x + b1)
    return _sigmoid(W2 @ h + b2)
```

Save and load weights as `.npz`:

```python
# Save
np.savez('model.npz', W1=W1, b1=b1, W2=W2, b2=b2)

# Load
data = np.load('model.npz')
W1, b1, W2, b2 = data['W1'], data['b1'], data['W2'], data['b2']
```

---

## Reference Implementation

The live implementation is at `behavior/personality/ml_personality.py`. Activate it in a scenario via:

```json
"personality": "ml:models/ship_brain.npz"
```

```python
# behavior/personality/ml_personality.py
import numpy as np
from behavior.personality import Personality, ShipState, state_to_vector, MODULE_NAMES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class MLPersonality(Personality):
    """
    Two-layer MLP personality loaded from a .npz weights file.
    Architecture: Linear(20, H) → Tanh → Linear(H, 6) → Sigmoid.
    """

    def __init__(self, weights_path: str) -> None:
        data    = np.load(weights_path)
        self.W1 = data['W1']   # (H, 20)
        self.b1 = data['b1']   # (H,)
        self.W2 = data['W2']   # (6, H)
        self.b2 = data['b2']   # (6,)

    def get_weights(self, state: ShipState) -> dict[str, float]:
        x   = state_to_vector(state)          # (20,)
        h   = np.tanh(self.W1 @ x + self.b1)  # (H,)
        out = _sigmoid(self.W2 @ h + self.b2)  # (6,)
        return dict(zip(MODULE_NAMES, out.tolist()))
```

---

## Training Approaches

### Option A — Supervised imitation of ChatGPT labels (current approach)

The pipeline generates diverse battle scenarios, runs headless simulations to capture ship state snapshots, sends each snapshot to ChatGPT with a structured prompt, and trains the MLP on `(state_vector, chatgpt_weights)` pairs.

**Run commands (from project root):**

```bash
python3 generate_scenarios.py --count 10
python3 collect_data.py --scenarios wargames_2d/scenarios/generated/
python3 label_with_chatgpt.py --unlabeled-dir data/unlabeled --stratified 100
python3 train_model.py
```

Stratified sampling (`--stratified N`) distributes labels proportionally across 6 tactical bins (weapon_loaded, hp level, has_target, can_fire, outnumbered, inside_objective) to avoid over-representing any single game state. Use `source .env` before the label step for `OPENAI_API_KEY`.

**Expected training time:** seconds to a few minutes on Apple Silicon with MLX, or CPU with numpy.

---

### Option B — Supervised imitation of the rule-based personality (fastest baseline)

Collect `(ShipState, target_weights)` pairs by running the simulation with `DefaultPersonality` and logging every call to `get_weights`. Train the network to minimize MSE between its output and the rule-based output.

This gives a smooth, generalizing approximation of the existing rules. The `LoggingPersonality` wrapper in `behavior/personality/logging_personality.py` handles data collection:

```python
logger = LoggingPersonality(DefaultPersonality())
for ship in ShipRegistry.ships:
    ship.brain.personality = logger

run_scenario(config)

states, weights = logger.to_arrays()  # (N, 20) and (N, 6)
```

After a few minutes of simulation you will have tens of thousands of samples. Shuffle, split 80/20, train with Adam and MSE loss.

---

### Option C — Neuroevolution (no gradient, no reward shaping)

Run N simulation episodes, score each by a fitness function, keep the best networks, mutate and repeat. No backpropagation required — works entirely with numpy.

**Why this is attractive here:**
- No reward signal design needed beyond a win condition
- The genome is just 870 floats — mutation and crossover are trivial
- Easy to parallelise: each episode is independent
- The simulation already runs headlessly

**Fitness candidates:**
- `kills / max(deaths, 1)` — K/D ratio over the episode
- `(faction_ships_remaining / initial_ships) - (enemy_ships_remaining / initial_ships)` — net fleet advantage at episode end
- `seconds_inside_objective / episode_length` — time-on-target, if territorial control matters

**Minimal loop:**

```python
import numpy as np
import copy, random

def random_genome(hidden=32):
    return {
        'W1': np.random.randn(hidden, 20) * 0.1,
        'b1': np.zeros(hidden),
        'W2': np.random.randn(6, hidden) * 0.1,
        'b2': np.zeros(6),
    }

def mutate(genome, sigma=0.02):
    g = copy.deepcopy(genome)
    for k in g:
        g[k] += np.random.randn(*g[k].shape) * sigma
    return g

def evolve(population, fitnesses, elite=4, children=16):
    ranked    = sorted(zip(fitnesses, population), reverse=True)
    survivors = [g for _, g in ranked[:elite]]
    offspring = [mutate(random.choice(survivors)) for _ in range(children)]
    return survivors + offspring
```

---

### Option D — Policy gradient / PPO (most capable)

Standard deep RL. The network is the policy; rewards are per-step signals from the simulation.

**Prerequisites not yet in the codebase:**

`ShipState` does not currently expose per-frame damage events. To compute a per-step reward you need two counters on `Ship`:

```python
# entities/ship.py  — add to __init__
self.damage_dealt_this_frame    = 0.0
self.damage_received_this_frame = 0.0
```

Reset both to `0.0` at the start of each frame, accumulate in `apply_damage` and `Weapon._fire_single_shot`, then expose them in `build_ship_state` as additional `ShipState` fields.

**Reward signal suggestion:**

```
r_t = damage_dealt_this_frame
    - damage_received_this_frame * 0.5
    + 0.1 * float(inside_objective)
    - 5.0 * float(died_this_frame)
```

**Credit assignment:** with a 6-second reload cycle the effective time horizon is ~360 frames at 60 fps. A discount factor of `γ = 0.995` gives a half-life of ~138 frames (~2.3 seconds), which is within one reload cycle.

**Framework:** PyTorch or JAX for training; export to numpy `.npz` for deployment. The inference path (`MLPersonality.get_weights`) has no framework dependency.

---

## Headless Simulation for Training

The PyGame render loop is the only thing that requires a display. To run episodes purely for data collection or fitness evaluation:

```python
# headless_sim.py
import pygame
pygame.init()   # needed for Vector2; does not open a window

from registry import ShipRegistry
from spawner import spawn_faction
from obstacle_spawner import spawn_obstacles
from objective import Objective

def run_episode(personality_f0, personality_f1, max_steps=3600) -> dict:
    ShipRegistry.clear()

    f0_center = pygame.math.Vector2(-400, 0)
    f1_center = pygame.math.Vector2( 400, 0)
    obstacles  = spawn_obstacles(seed=42)
    obj_f0     = Objective(position=f1_center, patrol_radius=200.0)
    obj_f1     = Objective(position=f0_center, patrol_radius=200.0)

    spawn_faction(30, 0, f0_center, obj_f0, personality=personality_f0)
    spawn_faction(30, 1, f1_center, obj_f1, personality=personality_f1)

    dt = 1.0 / 60.0
    for _ in range(max_steps):
        snapshot = list(ShipRegistry.ships)
        for ship in snapshot:
            if ship.is_alive() and ship.brain:
                ship.brain.update(ship, dt, ShipRegistry)
        for ship in snapshot:
            if ship.is_alive() and ship.weapon:
                ship.weapon.update(ship, dt, ShipRegistry)
        for ship in ShipRegistry.ships:
            ship.update_target(ShipRegistry)

        if not ShipRegistry.ships_by_faction.get(0) \
        or not ShipRegistry.ships_by_faction.get(1):
            break

    n0 = len(ShipRegistry.ships_by_faction.get(0, []))
    n1 = len(ShipRegistry.ships_by_faction.get(1, []))
    return {'faction0_remaining': n0, 'faction1_remaining': n1}
```

`pygame.init()` without `pygame.display.set_mode()` runs without opening a window. The simulation is otherwise identical to the interactive version.

---

## Evaluation

### Comparing a trained personality against the default

Run `N` episodes (recommended: 100) with the ML personality on one faction and `DefaultPersonality` on the other, alternating sides. Report:

- Win rate (fleet still has ships when the other does not)
- Average ships remaining at episode end
- Average episode length (shorter = more decisive)

Because neuroevolution and RL can overfit to a specific opponent, always evaluate against multiple opponents including random weight vectors and static (weight=constant) baselines.

### Using the in-game overlay

During interactive play, click any ship running the ML personality. The weight overlay shows `get_weights` output each frame (smoothed slightly for display). This is the fastest way to sanity-check that the model is producing sensible values:

- All outputs near 0.5 every frame → the model has not learned to differentiate states
- One output saturated at 1.0 always → likely a bias issue in training
- `BoundsModule` or `CollisionAvoidanceModule` near 0 → the model is suppressing safety modules, watch for ships flying into walls

---

## Extending ShipState

If your model needs features not currently in `ShipState`, add them to `build_ship_state` in `behavior/personality/__init__.py`. Keep the dataclass `frozen=True`. Do not expose mutable references — only scalars, booleans, and plain floats.

Adding a field to `ShipState` is a breaking change for any serialized training data — regenerate `data/unlabeled/` and `data/labeled/` after any vector change.

Candidates that could improve model quality:

| Feature | How to compute |
|---|---|
| `friend_hp_mean` | mean `hp_frac` of nearby friendlies |
| `enemy_closing_count` | enemies where `closing_speed > 0` |
| `time_since_last_kill` | counter on ship, reset on target death |
| `damage_dealt_this_frame` | running accumulator on ship (needed for RL reward) |
| `damage_received_this_frame` | running accumulator on ship (needed for RL reward) |
| `objective_friend_count` | friends currently `inside_objective` |

---

## File Conventions

```
behavior/
  personality/
    __init__.py               # ShipState, build_ship_state, state_to_vector,
                              # MODULE_NAMES, Personality ABC, DefaultPersonality
    ml_personality.py         # MLPersonality — live reference implementation
    logging_personality.py    # LoggingPersonality — wraps any personality for data collection
    <your_personality>.py     # one file per trained variant

models/
  ship_brain.npz              # default trained model (W1, b1, W2, b2)
  <run_id>_v<n>.npz           # versioned weight archives
  <run_id>_training_log.json  # loss curves, hyperparams

data/
  unlabeled/                  # raw state snapshots from collect_data.py
  labeled/                    # ChatGPT-labeled (state, weights) pairs

# Project root scripts
generate_scenarios.py         # generate diverse scenario JSON files
collect_data.py               # run scenarios headlessly, capture state snapshots
label_with_chatgpt.py         # send snapshots to ChatGPT, write labels.jsonl
train_model.py                # train MLP on labels, export models/ship_brain.npz
```
