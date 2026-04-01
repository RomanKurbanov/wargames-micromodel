---
name: strategy-game-simulation
description: Helps an agent work on strategy-game and tactical-simulation codebases, especially 2D boid or steering-based ship simulations. Use when implementing or debugging game logic, scenario generation, AI-controlled agents, vector blending, tactical priorities, visualization, telemetry, or behavior demos for autonomous units in a simulation or RTS-like sandbox.
compatibility: Designed for code agents working on game/simulation repositories; useful for Claude Code or similar tools.
metadata:
  author: openai
  version: "1.0"
---

# Strategy Game Simulation

Use this skill when the repository is a strategy game, tactical sandbox, boid simulation, steering-behavior demo, or autonomous-agent battlefield prototype.

## What this skill is for

This skill helps you act like a practical game developer working on a simulation-heavy strategy project.

Focus on:
- gameplay readability
- deterministic simulation behavior
- tactical clarity
- useful debugging tools
- scenario-driven testing
- believable AI behavior
- implementation choices that support demos, iteration, and tuning

Do not optimize for abstract elegance at the cost of visible behavior.

## Core mindset

Treat the project as a game/simulation first, and an ML project second.

Your job is to help build systems that:
- are inspectable
- produce readable outcomes
- support fast iteration
- are easy to tune
- can be demonstrated visually

When proposing changes, prefer solutions that make the simulation easier to understand on screen.

## Default assumptions

Unless the code clearly indicates otherwise, assume:
- the game is real-time or tick-based
- the world is 2D top-down
- entities are autonomous units with steering or movement logic
- behavior should be explainable in terms of priorities, forces, vectors, or state
- scenarios should be reproducible from files or seeds
- visual debugging is important
- the repository may include offline Python tooling and a separate runtime in another language

## What to prioritize

1. Simulation stability
2. Clear control abstractions
3. Observable runtime state
4. Scenario reproducibility
5. Debuggable behavior
6. Visual clarity of outcomes
7. Only then model sophistication

If there is a choice between a clever system and one that is easier to demonstrate and debug, prefer the latter.

## Working rules

### 1. Preserve gameplay-facing abstractions
If the codebase already has high-level controls like:
- flocking
- evasion
- attack
- objective
- formation
- caution
- aggression

prefer to work at that level instead of exposing raw low-level knobs unless there is a concrete need.

### 2. Avoid overfitting to implementation details
A simulation may internally compute many vectors, forces, and sub-weights. Do not assume the model or controller should directly own all of them. It is often better to let code keep low-level composition and let external systems adjust high-level priorities.

### 3. Make behavior legible
When adding or changing features, ask:
- can a developer see why a unit moved this way?
- can the dominant influences be displayed?
- can the same scenario be replayed?

If not, improve observability.

### 4. Prefer curated scenarios over large vague demos
For demonstrations, short handcrafted scenarios are usually better than chaotic large battles.

### 5. Use hard rules for safety-critical constraints
Collision avoidance, map bounds, impossible actions, and physical constraints should usually remain enforced in code even if ML or dynamic weighting is added.

### 6. Expect stateless controllers to need inertia
If a controller updates every frame from current state only, expect oscillation. Suggest smoothing, hysteresis, or limited state memory before proposing more complex sequence models.

## When implementing features

### Simulation code
Prefer:
- fixed-timestep or clearly defined update ordering
- pure or mostly pure calculations for steering and evaluation
- deterministic scenario loading
- explicit entity state structs/classes
- debug-friendly logging

### Scenario systems
Prefer scenario files that define:
- entities
- teams
- positions
- headings
- objectives
- obstacles
- scripted parameters
- optional expected outcomes

### Visualization
Recommend overlays for:
- selected-unit state
- current priorities/weights
- steering vectors
- danger or collision indicators
- target or objective markers
- team coloring

### AI/behavior systems
Prefer:
- high-level tactical priorities driving existing movement systems
- simple interfaces between simulation and model/tooling
- small, fixed schemas for features and outputs
- smoothing between predicted and applied values

## If the repo includes ML or external control

Treat ML as a behavior-modulation layer unless the code clearly expects more.

Good patterns:
- runtime model outputs a few tactical priorities
- game code maps those priorities to movement behavior
- scenario generator produces reproducible states
- debug UI shows predicted vs applied values

Be skeptical of:
- huge output spaces with weak semantics
- direct control of too many low-level parameters
- model integration that reduces explainability
- runtime dependencies that make the demo fragile

## Debugging checklist

When behavior looks wrong, inspect in this order:
1. scenario correctness
2. feature extraction correctness
3. normalization / scaling issues
4. update ordering
5. smoothing / hysteresis
6. rule overrides
7. visualization mismatch
8. only then model quality

## Expected deliverables this skill should encourage

Good contributions include:
- cleaner scenario formats
- debug overlays
- telemetry/logging
- selected-unit inspection panels
- deterministic replay support
- clear feature schemas
- behavior comparison tools
- side-by-side baseline vs adaptive runs

## Anti-patterns

Avoid pushing the project toward:
- full production architecture too early
- excessive abstraction layers
- hidden state mutations without visibility
- flashy visuals that obscure unit intent
- AI systems that cannot be explained to the developer
- long demos with no clear before/after comparison

## Response style for this project

When helping on this repository:
- be concrete
- think like a gameplay programmer
- suggest visible ways to verify behavior
- propose implementation steps in repository terms
- prefer small changes that improve demonstration quality

For deeper design guidance, see:
- references/REFERENCE.md