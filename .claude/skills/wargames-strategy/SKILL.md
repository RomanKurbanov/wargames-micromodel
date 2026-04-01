---
name: wargames-strategy
description: RTS strategy knowledge for wargames scenario design, synthetic data evaluation, and weight assessment. Use when the user asks about scenario generation, evaluating ship state features, checking if AI weights make strategic sense, or designing asymmetric faction configurations. This skill encodes classical RTS principles including supply lines, chokepoints, flanking routes, objective control, and rock-paper-scissors weapon dynamics.
---

# Wargames Strategy Skill

This skill provides strategic reasoning guidance for autonomous ship simulation in wargames scenarios. It helps evaluate whether scenarios create meaningful strategic choices and whether AI decisions reflect sound tactical reasoning.

## Core RTS Principles

### Scenario Design

**Asymmetry should create strategic depth, not just stat differences:**
- Consider rock-paper-scissors dynamics: fast swarms vs slow snipers vs durable tanks
- Weapon range asymmetry creates engagement distance strategies
- Ship count asymmetry should scale with stat differences (fewer high-stat ships vs more low-stat ships)
- Ensure both factions have viable paths to victory

**Map layout considerations:**
- Spawn safety: ships should have time to react after spawning
- Chokepoints: obstacles create natural bottlenecks that favor certain tactics
- Flanking routes: multiple paths to objectives prevent stalemates
- Objective accessibility: objectives should be reachable without excessive risk

**Scenario evaluation checklist:**
- [ ] Can ships exit spawn safely without immediate engagement?
- [ ] Are there multiple routes to objectives (avoid single chokepoint deathmatches)?
- [ ] Does obstacle density create meaningful cover or just visual clutter?
- [ ] Are spawn distances balanced so neither faction has unfair positioning advantage?
- [ ] Do weapon ranges create interesting engagement windows?

### Synthetic Data Evaluation

**Ship state feature completeness:**
When evaluating captured ship states for training data, check that:
- All 15 ShipState features are present and normalized to [0, 1]
- Special cases handled: `math.inf` distances normalize to 0.0
- Temporal consistency: consecutive snapshots show coherent state evolution
- Edge cases captured: reloading states, low HP, isolated ships, group engagements

**Data quality indicators:**
- Diversity: states span early game, mid game, and late game scenarios
- Balance: both winning and losing positions are represented
- Edge cases: include isolated ships, surrounded ships, ships with clear objectives
- Temporal coherence: state transitions between frames are physically plausible

### Weight Assessment

**Strategic reasoning validation:**

| Ship State | Expected Weight Pattern | Strategic Rationale |
|------------|------------------------|---------------------|
| Low HP (< 30%) | High Evasion, Low Objective | Survive first, objectives second |
| Reloading | High Evasion, Low Combat Modules | Retreat to safety while vulnerable |
| Loaded + Close Target | High FiringSolution, Low Evasion | Commit to engagement |
| Outnumbered (ratio < 0.6) | High Cohesion/Alignment | Tighten formation for mutual support |
| Inside Objective | High Objective, Moderate Combat | Hold position while engaging |
| High HP + Loaded | Aggressive FiringSolution | Push for advantage |

**Red flags in weight outputs:**
- Weights near 0.5 for all modules = model not learning
- BoundsModule or CollisionAvoidance near 0 = ships flying into walls
- Evasion always high = model afraid to engage
- FiringSolution always high = model ignores reload state

**Weight sanity checks:**
- Reloading ships should suppress FiringSolutionModule
- Low HP should increase EvasionModule weight
- ObjectiveModule should be active when objective is approachable
- BoundsModule should rarely drop below 0.5 (safety critical)

## Example Workflows

### Evaluating a New Scenario

```
User: "Does this asymmetric scenario make strategic sense?"

Skill applies:
1. Check spawn safety (distance between spawn centers vs weapon ranges)
2. Identify chokepoints (obstacle density along paths)
3. Assess faction asymmetry (do stat differences create meaningful choices?)
4. Verify objective accessibility (can both factions reach objectives?)
```

### Assessing Training Data

```
User: "Here are 100 ship state snapshots - are they good training data?"

Skill applies:
1. Verify all 15 features normalized to [0, 1]
2. Check temporal coherence across frames
3. Assess diversity (early/mid/late game coverage)
4. Flag edge cases (isolated ships, surrounded ships)
```

### Reviewing AI Weights

```
User: "These weights look wrong - what should they be?"

Skill applies strategic reasoning:
1. Analyze ship state (HP, ammo, target distance, friend count)
2. Determine tactical situation (retreat, engage, hold, flank)
3. Recommend weight adjustments for coherent behavior
```

## Integration with ML Pipeline

This skill supports the micro-model distillation workflow:

1. **Scenario generation**: Use RTS principles to create diverse, strategically interesting scenarios
2. **Data validation**: Ensure captured ship states are complete, normalized, and diverse
3. **Weight verification**: Check that ChatGPT outputs (and later, micro-model outputs) reflect sound strategic reasoning
4. **Evaluation**: Compare model performance against baseline using win rates and strategic quality

## When NOT to Use This Skill

- Simple debugging of simulation physics (use ship state inspection instead)
- Rendering issues (use main.py debug output)
- Basic Python syntax questions (use standard coding skills)
- When the user explicitly asks for non-strategic analysis
