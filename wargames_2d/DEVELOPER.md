# Wargames 2D — Developer Guide

A 2D autonomous fleet battle sandbox built with Python and PyGame. Two factions of AI-driven ships fight over area objectives using a modular steering system. There is no player unit — it is a pure simulation.

## Quick Start

```bash
cd wargames_2d
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

**Runtime controls:** WASD to pan camera · left-click a ship to inspect it · ESC to quit.

---

## Architecture Overview

```
main.py                     # PyGame loop, rendering, camera, ship selection
registry.py                 # Global object/ship lists (ShipRegistry)
objective.py                # Shared team goal (area or target-ship)
spawner.py                  # Faction setup: ships, weapons, brains
obstacle_spawner.py         # Poisson-disk asteroid placement

entities/
  object.py                 # Base entity: position, size, auto-register
  ship.py                   # Ship stats, combat, target acquisition
  weapon.py                 # Cassette fire model, LoS check, tracers
  obstacle.py               # Static world geometry

behavior/
  steering_module.py        # SteeringModule ABC + safe_normalize()
  ship_brain.py             # ShipBrain: weight aggregator + flight model
  personality.py            # ShipState snapshot + Personality ABC + DefaultPersonality
  navigate/                 # Objective, bounds, collision, separation
  flock/                    # Alignment, cohesion (friendly neighbors)
  engage/                   # Evasion, firing solution (enemy neighbors)

factions.py                 # faction int → RGB color
```

### Update order each frame

```
1. ShipBrain.update()       — steering modules → _aviate() → new position
2. Weapon.update()          — fire/reload state machine, tracer decay
3. Ship.update_target()     — nearest visible enemy scan
```

---

## Core Concepts

### The Steering Pipeline

Each ship carries a `ShipBrain` that holds an ordered list of `SteeringModule` instances. Every frame:

1. `compute_weight_overrides(ship, registry)` is called — returns `{ClassName: float}` overrides.
2. For each module, the effective weight is `overrides.get(ClassName, module.weight)`.
3. Weights below zero or non-finite are silently ignored (safety guard).
4. All `module.compute_steering(ship, registry)` results are scaled by their effective weight and summed into a single steering vector.
5. The summed vector is passed to `_aviate()` — the flight model that translates it into velocity and position.

The steering vector's **direction** determines where the ship wants to go. Its **magnitude** (clamped to 1.0) controls how hard the ship drives: magnitude 1 = full throttle, 0 = coast to a stop.

### ShipRegistry

`ShipRegistry` is a class-level (effectively global) store. All entities register themselves on construction and unregister on destruction.

| Attribute | Type | Contents |
|---|---|---|
| `objects` | `list[Object]` | Every live entity including ships and obstacles |
| `ships` | `list[Ship]` | Every live ship |
| `ships_by_faction` | `dict[int, list[Ship]]` | Ships grouped by `faction` int |

Any module can query these freely during `compute_steering`.

### Object and Ship Lifecycle

`Object.__init__` registers in `objects`. `Ship.__init__` also registers in `ships` and `ships_by_faction`. `Ship.destroy()` sets `hp = 0` and unregisters from both lists. After `destroy()`, `is_alive()` always returns `False` and the ship is invisible to all registry queries.

### Objective

An `Objective` is a shared navigational goal for an entire faction.

```python
# Static area — ships orbit the position
obj = Objective(position=Vector2(400, 0), patrol_radius=200.0)

# Enemy ship — ships chase it; falls back to last position on death
obj = Objective(target_ship=some_ship, patrol_radius=150.0)
```

`obj.position` always returns the current world-space center (or `None` if invalid). `obj.is_valid()` is a safe guard before use. Assigned to each ship as `ship.objective`.

---

## Writing a Steering Module

Subclass `SteeringModule` and implement `compute_steering`. Place the file anywhere under `behavior/`.

```python
# behavior/engage/my_module.py
import pygame
from behavior.steering_module import SteeringModule, safe_normalize

class MyModule(SteeringModule):
    weight = 0.5   # default weight, overridable per-frame

    def __init__(self, some_param: float = 100.0):
        self.some_param = some_param

    def compute_steering(self, ship, registry) -> pygame.math.Vector2:
        # Return a Vector2. Direction = desired heading. Magnitude = desired effort [0..1].
        # Return zero to abstain.
        ...
```

**Conventions:**
- Return a **unit vector** (or shorter) unless you want to amplify — the brain sums all modules before clamping.
- Use `safe_normalize(v)` instead of `v.normalize()` to avoid crashes on zero-length vectors.
- Read but never write `ship` or `registry` state. Side effects during steering break the update order.
- Keep `weight` as a class attribute so it shows up in the overlay and is accessible to `compute_weight_overrides` without an instance.

**Base classes** for common neighbor queries:

| Base class | Method provided | Typical use |
|---|---|---|
| `FriendlyBaseModule(neighbor_radius)` | `get_neighbors(ship, registry)` — same-faction ships in radius | Flock behavior |
| `EnemyBaseModule(detection_range)` | `get_enemies(ship, registry)` — enemy ships in range | Combat behavior |

### Registering the Module

Add an import and instance in `spawner._make_brain()`:

```python
from behavior.engage.my_module import MyModule

def _make_brain() -> ShipBrain:
    return ShipBrain(modules=[
        ...
        MyModule(some_param=200.0),
    ])
```

Add its display name to `_MODULE_LABELS` in `main.py` to show it in the weight overlay:

```python
_MODULE_LABELS = {
    ...
    'MyModule': 'MyMod',
}
```

---

## Personality — Dynamic Weight Generation

`Personality` is the sole extension point for changing how ships behave. It answers one question per frame: *"given what this ship can observe, how strongly should each steering module pull?"*

### Key constraint

A personality **only produces weights**. It never receives a reference to any live game object — only a frozen `ShipState` snapshot. It cannot call methods, mutate state, or influence vector calculations. This is enforced structurally, not by convention.

```
Live game objects  →  build_ship_state()  →  ShipState (frozen)
                                                    ↓
                                          personality.get_weights()
                                                    ↓
                                          {ClassName: float}  →  ShipBrain sums modules
```

### Writing a personality

```python
# behavior/personality/aggressive.py
from behavior.personality import Personality, ShipState

class AggressivePersonality(Personality):
    def get_weights(self, state: ShipState) -> dict[str, float]:
        if not state.weapon_loaded:
            return {
                'EvasionModule':        0.9,
                'FiringSolutionModule': 0.0,
                'ObjectiveModule':      0.0,
            }
        # Push hard when in range, ignore evasion
        return {
            'FiringSolutionModule': 1.0,
            'EvasionModule':        0.1,
            'ObjectiveModule':      0.2,
        }
```

Inject it at spawn time — one instance can be **shared across an entire faction**:

```python
from behavior.personality.aggressive import AggressivePersonality

personality = AggressivePersonality()
spawn_faction(30, 0, f0_center, obj_f0, personality=personality)
```

Or assign per-ship after spawning:

```python
ships = spawn_faction(30, 0, f0_center, obj_f0)
for ship in ships[:10]:
    ship.brain.personality = ScoutPersonality()
for ship in ships[10:]:
    ship.brain.personality = BruiserPersonality()
```

### ShipState reference

`build_ship_state()` is called once per ship per frame before `get_weights()`. All values are pre-normalised — personalities don't need to know internal game constants.

**Health**

| Field | Type | Range | Meaning |
|---|---|---|---|
| `hp_frac` | float | 0–1 | Current HP / max HP |
| `shield_frac` | float | 0–1 | Current shield / max shield |

**Weapon**

| Field | Type | Meaning |
|---|---|---|
| `weapon_loaded` | bool | `True` = cassette ready, `False` = reloading |
| `ammo_frac` | float 0–1 | Shots remaining in cassette / cassette size |
| `reload_progress` | float 0–1 | 0 = reload just started, 1 = complete |
| `can_fire` | bool | Weapon is in range / arc / LoS of current target right now |

**Motion**

| Field | Type | Meaning |
|---|---|---|
| `speed_frac` | float 0–1 | Current speed / max speed |

**Target**

| Field | Type | Meaning |
|---|---|---|
| `has_target` | bool | A locked enemy exists |
| `target_distance` | float | Pixels to target; `math.inf` if none |
| `target_hp_frac` | float 0–1 | Target's HP fraction; 1.0 if no target |
| `closing_speed` | float | px/s, positive = closing in; 0 if no target |

**Team (fleet-wide)**

| Field | Type | Meaning |
|---|---|---|
| `friend_count` | int | Alive friendlies excluding self |
| `enemy_count` | int | Alive enemies across all factions |

**Objective**

| Field | Type | Meaning |
|---|---|---|
| `objective_distance` | float | Pixels to objective centre; `math.inf` if none |
| `inside_objective` | bool | Ship is inside the patrol ring |

### Example: situational personality

```python
class TacticalPersonality(Personality):
    def get_weights(self, state: ShipState) -> dict[str, float]:
        overrides = {}

        # --- Ammo state ---
        if not state.weapon_loaded:
            overrides['EvasionModule']        = 0.9
            overrides['FiringSolutionModule'] = 0.0
            overrides['ObjectiveModule']      = 0.0
            overrides['AlignmentModule']      = 0.05
            overrides['CohesionModule']       = 0.05
            return overrides

        overrides['FiringSolutionModule'] = 1.0
        overrides['EvasionModule']        = 0.3

        # --- Hull damage: retreat when badly hurt ---
        if state.hp_frac < 0.3:
            retreat = 1.0 - state.hp_frac / 0.3
            overrides['EvasionModule']        = min(1.0, 0.3 + retreat * 0.5)
            overrides['FiringSolutionModule'] = max(0.0, 1.0 - retreat * 0.8)

        # --- Outnumbered: tighten up ---
        ratio = state.friend_count / max(state.enemy_count, 1)
        if ratio < 0.6:
            overrides['CohesionModule']   = 0.4
            overrides['AlignmentModule']  = 0.35

        # --- Already in a firing solution: hold position ---
        if state.can_fire:
            overrides['FiringSolutionModule'] = 0.0   # don't need to close further
            overrides['ObjectiveModule']      = 0.1

        return overrides
```

### Weight safety

`ShipBrain.update()` automatically:
- Drops any weight that is `NaN` or `Inf`
- Floors weights at `0.0` (negative values are treated as zero)

Return clean values in [0, 1] for clarity.

---

## Weapon Line-of-Sight

`Weapon.can_fire_at(ship, target_pos, registry)` checks three things in order:

1. **Range** — distance to target ≤ `weapon.range`
2. **Angle** — bearing to target within `±max_fire_angle` degrees of heading
3. **LoS** — no object in `registry.objects` (obstacle or ship, excluding shooter and target) intersects the segment from shooter to target

The LoS test uses a segment-circle intersection (`_segment_circle_intersects`) iterated over all registered objects. Obstacles and ships both block shots because they are both subclasses of `Object` and auto-register.

To make an entity transparent to shots, override or filter `registry.objects` — or add a `blocks_los: bool = True` flag to `Object` and check it in `Weapon.can_fire_at`.

---

## Adding New Entity Types

Extend `Object` and call `super().__init__(position, size)`. The entity is then:
- Included in collision avoidance and LoS checks automatically
- Excluded from flock/target queries (those filter on `ShipRegistry.ships`)

```python
# entities/supply_depot.py
from entities.object import Object

class SupplyDepot(Object):
    def __init__(self, position, size=30.0):
        super().__init__(position, size)
        self.resupply_rate = 5.0   # HP/sec to nearby ships
```

To participate in targeting or faction logic, also register via `ShipRegistry.register_ship` / `ships_by_faction` (see `Ship.__init__` for the pattern).

---

## Tuning Reference

### Ship defaults (`entities/ship.py`)

| Parameter | Default | Effect |
|---|---|---|
| `max_speed` | 100 px/s | Top speed |
| `forward_acceleration` | 40 px/s² | Thrust |
| `brake_acceleration` | 200 px/s² | Drag/braking |
| `turn_rate` | 180 °/s | Rotation speed |
| `lateral_damping` | 3.0 | Exponential sideways drag coefficient |
| `hp` | 100 | Hull points |
| `shield` | 200 | Shield absorbs damage first |
| `visual_range` | 800 px | Target acquisition range |

### Weapon defaults (`entities/weapon.py`)

| Parameter | Default | Effect |
|---|---|---|
| `range` | 400 px | Maximum fire distance |
| `damage` | 10.0 | Damage per shot |
| `fire_rate` | 5 shots/s | Rate within a cassette |
| `max_fire_angle` | 25° | Half-cone firing arc |
| `cassette_size` | 5 | Shots before reload |
| `reload_duration` | 6.0 s | Full reload time |

### Module static weights (`behavior/`)

| Module | Default weight | Role |
|---|---|---|
| `ObjectiveModule` | 0.8 | Pull toward team objective |
| `BoundsModule` | 1.0 | Keep inside world boundary |
| `CollisionAvoidanceModule` | 0.9 | Avoid all objects |
| `SeparationModule` | 0.5 | Personal space from all ships |
| `AlignmentModule` | 0.3 | Match velocity of loaded flockmates |
| `CohesionModule` | 0.15 | Stay close to flockmates |
| `EvasionModule` | 0.6 | Avoid enemy weapon range |
| `FiringSolutionModule` | 0.8 | Close to firing range |

---

## Debugging a Running Simulation

### Ship selection overlay

Click any ship in the window. A panel appears in the top-right corner showing:
- The effective weight of each module **this frame** (static weight or override)
- A fill bar (0–1 range, color-coded dark-red → green)
- A numeric readout

Weights are **exponentially smoothed** before display (time constant ≈ 0.12 s) so short-lived spikes show up as a fast flash rather than noise.

The smoothed values reflect the *last computed* override, not what the brain computed mid-flight. If you are adding a new override rule, select a ship under those conditions and watch the relevant bars.

### Console tracing

The cleanest approach is to add per-frame prints guarded by a ship identity check in `compute_weight_overrides` or inside a module's `compute_steering`. Since these run at 60 fps, throttle with a frame counter:

```python
# In ShipBrain.compute_weight_overrides — trace one specific ship
_TRACE_FRAME = [0]

def compute_weight_overrides(self, ship, registry):
    _TRACE_FRAME[0] += 1
    if _TRACE_FRAME[0] % 30 == 0:   # print every 0.5 s
        dist = (ship.target.position - ship.position).length() if ship.target else -1
        print(f"[{id(ship) & 0xFFFF:04x}] hp={ship.hp:.0f} "
              f"reload={ship.weapon.reloading} target_dist={dist:.0f}")
    ...
```

To trace a specific ship instance from `main.py`, store a reference:

```python
# In main(), after spawn_faction():
debug_ship = ShipRegistry.ships[0]

# In the game loop, before brain update:
if debug_ship.is_alive():
    print(debug_ship.velocity.length(), debug_ship.heading_angle)
```

### Drawing debug visuals

`main.py` renders after the physics update. Add any geometry before `pygame.display.flip()`:

```python
# Draw a ship's visual range circle
for ship in ShipRegistry.ships:
    sp = world_to_screen(ship.position, cam)
    pygame.draw.circle(screen, (60, 60, 60), sp, int(ship.visual_range), 1)

# Draw a line from a ship to its current target
for ship in ShipRegistry.ships:
    if ship.target:
        pygame.draw.line(screen, (200, 100, 0),
                         world_to_screen(ship.position, cam),
                         world_to_screen(ship.target.position, cam), 1)

# Draw the steering vector (scaled for visibility)
for ship in ShipRegistry.ships:
    if ship.brain:
        ov = ship.brain.compute_weight_overrides(ship, ShipRegistry)
        steer = pygame.math.Vector2(0, 0)
        for mod in ship.brain.modules:
            w = ov.get(type(mod).__name__, mod.weight)
            steer += mod.compute_steering(ship, ShipRegistry) * max(0.0, w)
        end = ship.position + steer * 40
        pygame.draw.line(screen, (0, 180, 255),
                         world_to_screen(ship.position, cam),
                         world_to_screen(end, cam), 2)
```

### Headless / unit testing a module

You can test any module without running the full PyGame window. The only dependencies are `pygame.math.Vector2` (no display required) and a minimal stub for the registry:

```python
import pygame
pygame.init()   # required for Vector2 to work

from behavior.navigate.objective_module import ObjectiveModule
from objective import Objective

class _FakeShip:
    position  = pygame.math.Vector2(0, 0)
    velocity  = pygame.math.Vector2(0, 0)
    faction   = 0
    target    = None
    objective = Objective(position=pygame.math.Vector2(500, 0), patrol_radius=200)
    weapon    = None

class _FakeRegistry:
    ships = []; objects = []; ships_by_faction = {}

ship = _FakeShip()
mod  = ObjectiveModule(inner_radius=80.0)

result = mod.compute_steering(ship, _FakeRegistry())
print(result, result.length())   # should be close to (1, 0) and ~1.0
```

### Reproducing a specific layout

`obstacle_spawner.spawn_obstacles` accepts a `seed` parameter:

```python
obstacles = spawn_obstacles(seed=42)
```

Use this to freeze the obstacle layout while iterating on steering behavior.

---

## Extending the Simulation

### Multiple factions

`factions.py` supports up to four factions out of the box (green, red, blue, yellow). Add more by extending `COLORS`. Spawn additional factions in `main.py`:

```python
f2_center = pygame.math.Vector2(0, -500)
obj_f2 = Objective(position=f0_center, patrol_radius=200.0)
spawn_faction(20, 2, f2_center, objective=obj_f2)
```

Each faction's ships will automatically be included in enemy queries from all other factions.

### Per-ship brains

Each ship gets its own `ShipBrain` instance from `_make_brain()`. To give certain ships different behaviors, pass a different brain:

```python
ship.brain = ShipBrain(modules=[
    ObjectiveModule(),
    BoundsModule(),
    FiringSolutionModule(detection_range=800.0),
    # no flock modules — lone wolf
])
```

### Swapping personality at runtime

Personalities are plain attributes — replace them on a live brain at any time:

```python
ship.brain.personality = None                  # revert to static weights
ship.brain.personality = AggressivePersonality()  # hot-swap mid-battle
```

A single personality instance can be shared by any number of ships. Swapping on the shared instance changes all of them simultaneously — useful for faction-wide mode shifts.

### Making `course_lock` do something

`ship.course_lock` is currently a stub (`Vector2(0, 0)`). When non-zero, `_aviate` uses it as the facing direction independently of the movement direction — letting a ship strafe or keep guns on target while retreating. Set it in a module's `compute_steering` return value... actually set it directly before `_aviate` is called, or add a dedicated module that writes `ship.course_lock` as a side effect (be deliberate: this is the one sanctioned side-effect slot).

---

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
