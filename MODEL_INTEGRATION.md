# Ship-Brain Model — Integration Guide

This document explains how to use the exported ship-brain micro-model in any application.

## Overview

The model predicts per-ship tactical steering weights in real time.
Given a 20-float description of a ship's current situation, it outputs 4 weights (one per steering module) in the range [0, 1].

**Architecture:** `Linear(20, H) → Tanh → Linear(H, 4) → Sigmoid`
**Production model:** `models/ship_brain_v4.npz` (H = 64, 1604 parameters)

---

## Quick Start

### 1. Export the model

```bash
# JSON (works everywhere — no extra dependencies)
python export_model.py --model models/ship_brain_v4.npz --format json

# ONNX (Unity Sentis, ML.NET, onnxruntime-web, etc.)
pip install onnx
python export_model.py --model models/ship_brain_v4.npz --format onnx

# Both at once
python export_model.py --model models/ship_brain_v4.npz --format json --format onnx
```

Both outputs land in the current directory with the same stem as the input file
(e.g. `ship_brain_v4.json`, `ship_brain_v4.onnx`).

### 2. Run inference

```python
# Minimal Python — numpy only
import json, numpy as np

d   = json.load(open('ship_brain_v4.json'))
W1  = np.array(d['weights']['W1'], dtype=np.float32)   # (64, 20)
b1  = np.array(d['weights']['b1'], dtype=np.float32)   # (64,)
W2  = np.array(d['weights']['W2'], dtype=np.float32)   # (4, 64)
b2  = np.array(d['weights']['b2'], dtype=np.float32)   # (4,)

x   = build_input_vector(ship)       # see Section 2 below
h   = np.tanh(W1 @ x + b1)
out = 1.0 / (1.0 + np.exp(-np.clip(W2 @ h + b2, -20, 20)))

# out[0] = ObjectiveModule weight
# out[1] = FlockingModule weight
# out[2] = EvasionModule weight
# out[3] = AttackModule weight
```

---

## Section 1 — Forward Pass

```
x  (20,) float32
 └─ h  = tanh( W1 @ x + b1 )     W1: (hidden, 20),  b1: (hidden,)
     └─ out = sigmoid( W2 @ h + b2 )  W2: (4, hidden),   b2: (4,)
```

| Tensor | Shape | Description |
|--------|-------|-------------|
| `W1`   | `(hidden, 20)` | First layer weights |
| `b1`   | `(hidden,)`    | First layer biases |
| `W2`   | `(4, hidden)`  | Second layer weights |
| `b2`   | `(4,)`         | Second layer biases |

`hidden` = 64 for the current production model.

**Sigmoid overflow guard** — clip logits to [-20, 20] before applying sigmoid to avoid numerical issues in languages without native overflow protection.

---

## Section 2 — Building the 20-Float Input Vector

Build this vector fresh each simulation tick from the ship's current game state.
The order and normalization **must** exactly match the table below.

### Normalization constants

| Constant | Value | Used for |
|----------|-------|----------|
| `DIST_SCALE`       | 400.0 px   | target and objective distance decay |
| `SPEED_SCALE`      | 100.0 px/s | closing-speed tanh scale |
| `COUNT_SCALE`      | 30.0 ships | friend and enemy count cap |
| `MAX_HP_REF`       | 265.0 HP   | ship HP profile stat |
| `MAX_SPEED_REF`    | 130.0 px/s | ship speed profile stat |
| `WEAPON_RANGE_REF` | 610.0 px   | weapon range profile stat |
| `WEAPON_DMG_REF`   |  38.0 dmg  | weapon damage profile stat |
| `RELOAD_DUR_REF`   |  11.0 s    | weapon reload duration profile stat |

### Feature table

| # | Name | Formula | Range | Edge case |
|---|------|---------|-------|-----------|
| 0 | `hp_frac` | `hp / max_hp` | [0, 1] | — |
| 1 | `shield_frac` | `shield / max_shield` | [0, 1] | `0.0` when ship has no shield |
| 2 | `weapon_loaded` | `float(not reloading)` | {0, 1} | `1.0` when no weapon |
| 3 | `ammo_frac` | `(cassette_size - shots_fired) / cassette_size` | [0, 1] | `1.0` when no weapon |
| 4 | `reload_progress` | `1 − reload_timer / reload_duration` while reloading; `1.0` when loaded | [0, 1] | `1.0` when no weapon |
| 5 | `speed_frac` | `velocity_magnitude / max_speed` | [0, 1] | — |
| 6 | `has_target` | `float(target != null && target.is_alive())` | {0, 1} | — |
| 7 | `target_distance` | `1.0 / (1.0 + dist / 400.0)` | (0, 1] | `0.0` when no target |
| 8 | `target_hp_frac` | `target.hp / target.max_hp` | [0, 1] | `1.0` when no target |
| 9 | `closing_speed` | `tanh( dot(ship_vel − target_vel, to_target_unit) / 100.0 )` | (−1, 1) | `0.0` when no target |
| 10 | `can_fire` | `float(loaded AND in_range AND in_arc AND clear_LoS)` | {0, 1} | `0.0` when no target or reloading |
| 11 | `friend_count` | `min(friends_in_visual_range / 30.0, 1.0)` | [0, 1] | — |
| 12 | `enemy_count` | `min(enemies_in_visual_range / 30.0, 1.0)` | [0, 1] | — |
| 13 | `objective_distance` | `1.0 / (1.0 + dist_to_objective / 400.0)` | (0, 1] | `0.0` when no objective |
| 14 | `inside_objective` | `float(dist_to_objective <= patrol_radius)` | {0, 1} | `0.0` when no objective |
| 15 | `max_hp` | `min(max_hp / 265.0, 1.0)` | [0, 1] | — |
| 16 | `max_speed_stat` | `min(max_speed / 130.0, 1.0)` | [0, 1] | — |
| 17 | `weapon_range` | `min(weapon_range / 610.0, 1.0)` | [0, 1] | `0.0` when no weapon |
| 18 | `weapon_damage` | `min(damage / 38.0, 1.0)` | [0, 1] | `0.0` when no weapon |
| 19 | `reload_duration` | `min(reload_duration / 11.0, 1.0)` | [0, 1] | `0.0` when no weapon |

### Feature notes

**Feature 7 & 13 — hyperbolic distance decay**
`1.0 / (1.0 + dist / 400.0)` maps:
- distance 0 px → 1.0 (very close)
- distance 400 px → 0.5
- distance ∞ (no target/objective) → 0.0

**Feature 9 — closing speed**
`closing_speed_raw = dot(ship_velocity − target_velocity, normalize(target_pos − ship_pos))`
Positive = ship is closing toward target. Negative = retreating.
Apply `tanh(v / 100.0)` to squash to (−1, 1).

**Features 11–12 — team counts**
Count ships within `visual_range` (default 800 px). No line-of-sight filtering.

**Features 15–19 — ship profile**
These are the ship's *configured* maximum stats, not current values. They are constant for a given ship and allow the model to adapt its strategy to the ship class (fast/fragile vs slow/tanky, long-reload sniper vs rapid-fire swarm, etc.).

---

## Section 3 — Outputs

| Index | Module | Meaning | Range |
|-------|--------|---------|-------|
| 0 | `ObjectiveModule` | Drive toward / hold the team objective area | [0, 1] |
| 1 | `FlockingModule` | Stay in formation with friendly ships | [0, 1] |
| 2 | `EvasionModule` | Dodge and retreat from incoming fire | [0, 1] |
| 3 | `AttackModule` | Manoeuvre to establish / hold a firing solution | [0, 1] |

Each output is a weight multiplier applied to the corresponding steering vector before summing.

**Excluded modules** (not predicted; use static defaults):
- `BoundsModule` — keeps ships in-bounds; can output magnitude up to 2.0
- `CollisionAvoidanceModule` — per-object avoidance; computed from per-object sizes

---

## Section 4 — Reference Implementations

### Python (numpy only)

```python
import math, numpy as np

# Normalization helpers
def norm_dist(d):
    return 1.0 / (1.0 + d / 400.0) if math.isfinite(d) else 0.0

def build_input_vector(
    hp_frac, shield_frac,
    weapon_loaded, ammo_frac, reload_progress,
    speed_frac,
    has_target, target_dist, target_hp_frac, closing_speed_raw, can_fire,
    friend_count, enemy_count,
    objective_dist, inside_objective,
    max_hp, max_speed, weapon_range, weapon_damage, reload_duration,
):
    return np.array([
        hp_frac,
        shield_frac,
        float(weapon_loaded),
        ammo_frac,
        reload_progress,
        speed_frac,
        float(has_target),
        norm_dist(target_dist),
        target_hp_frac,
        math.tanh(closing_speed_raw / 100.0),
        float(can_fire),
        min(friend_count / 30.0, 1.0),
        min(enemy_count  / 30.0, 1.0),
        norm_dist(objective_dist),
        float(inside_objective),
        min(max_hp        / 265.0, 1.0),
        min(max_speed     / 130.0, 1.0),
        min(weapon_range  / 610.0, 1.0),
        min(weapon_damage /  38.0, 1.0),
        min(reload_duration / 11.0, 1.0),
    ], dtype=np.float32)


def infer(W1, b1, W2, b2, x):
    h   = np.tanh(W1 @ x + b1)
    raw = W2 @ h + b2
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -20.0, 20.0)))
```

### Pseudocode (language-agnostic)

```
function normalize_input(ship, target, objective) -> float[20]:
    x[0]  = ship.hp / ship.max_hp
    x[1]  = ship.shield / ship.max_shield   // 0 if no shield
    x[2]  = ship.weapon_loaded ? 1.0 : 0.0
    x[3]  = (cassette_size - shots_fired) / cassette_size
    x[4]  = reloading ? (1 - reload_timer / reload_duration) : 1.0
    x[5]  = ship.speed / ship.max_speed
    x[6]  = target != null ? 1.0 : 0.0
    x[7]  = target != null ? 1 / (1 + distance(ship, target) / 400) : 0.0
    x[8]  = target != null ? target.hp / target.max_hp : 1.0
    x[9]  = target != null ? tanh(dot(ship.vel - target.vel, unit(target - ship)) / 100) : 0.0
    x[10] = can_fire(ship, target) ? 1.0 : 0.0
    x[11] = clamp(friend_count / 30.0, 0, 1)
    x[12] = clamp(enemy_count  / 30.0, 0, 1)
    x[13] = objective != null ? 1 / (1 + distance(ship, objective) / 400) : 0.0
    x[14] = inside_objective ? 1.0 : 0.0
    x[15] = clamp(ship.max_hp         / 265.0, 0, 1)
    x[16] = clamp(ship.max_speed      / 130.0, 0, 1)
    x[17] = clamp(ship.weapon_range   / 610.0, 0, 1)
    x[18] = clamp(ship.weapon_damage  /  38.0, 0, 1)
    x[19] = clamp(ship.reload_duration / 11.0, 0, 1)
    return x

function infer(W1, b1, W2, b2, x) -> float[4]:
    h   = tanh(matmul(W1, x) + b1)          // element-wise tanh
    raw = matmul(W2, h) + b2
    out = sigmoid(clamp(raw, -20, 20))       // element-wise sigmoid
    return out   // [objective, flocking, evasion, attack]
```

### C# / Unity (Sentis)

1. Export the ONNX file: `python export_model.py --model models/ship_brain_v4.npz --format onnx`
2. Add `ship_brain_v4.onnx` to your Unity `Assets/` folder.
3. Install the **Sentis** package via Package Manager.

```csharp
using Unity.Sentis;

public class ShipBrain : MonoBehaviour
{
    [SerializeField] ModelAsset modelAsset;

    IWorker worker;
    Model runtimeModel;

    void Awake()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(BackendType.GPUCompute, runtimeModel);
    }

    public float[] Infer(float[] inputVector20)
    {
        using var input = new TensorFloat(new TensorShape(1, 20), inputVector20);
        worker.Execute(input);
        using var output = worker.PeekOutput() as TensorFloat;
        output.MakeReadable();
        return new float[] {
            output[0, 0],   // ObjectiveModule
            output[0, 1],   // FlockingModule
            output[0, 2],   // EvasionModule
            output[0, 3],   // AttackModule
        };
    }

    void OnDestroy() => worker?.Dispose();
}
```

---

## Section 5 — Verification

After exporting, verify the JSON weights produce the same outputs as the original `.npz`:

```bash
python - <<'EOF'
import json, numpy as np

# Load exported JSON
d  = json.load(open('ship_brain_v4.json'))
W1 = np.array(d['weights']['W1'], dtype=np.float32)
b1 = np.array(d['weights']['b1'], dtype=np.float32)
W2 = np.array(d['weights']['W2'], dtype=np.float32)
b2 = np.array(d['weights']['b2'], dtype=np.float32)

# Load original .npz
ref = np.load('models/ship_brain_v4.npz')

assert np.allclose(W1, ref['W1']), "W1 mismatch"
assert np.allclose(b1, ref['b1']), "b1 mismatch"
assert np.allclose(W2, ref['W2']), "W2 mismatch"
assert np.allclose(b2, ref['b2']), "b2 mismatch"
print("Weights match — export is correct.")

# Smoke inference
x   = np.random.rand(20).astype(np.float32)
h   = np.tanh(W1 @ x + b1)
out = 1 / (1 + np.exp(-np.clip(W2 @ h + b2, -20, 20)))
print(f"Sample output: {out.round(3)}")
EOF
```

Verify the ONNX model (requires `pip install onnx onnxruntime`):

```bash
python - <<'EOF'
import onnx, onnxruntime as ort, numpy as np

onnx.checker.check_model(onnx.load('ship_brain_v4.onnx'))
sess = ort.InferenceSession('ship_brain_v4.onnx')
x    = np.random.rand(1, 20).astype(np.float32)
out  = sess.run(None, {'input': x})[0]
print(f"ONNX output: {out.round(3)}")
EOF
```
