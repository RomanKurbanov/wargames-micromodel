#!/usr/bin/env python3
"""
collect_data.py — Run headless scenario simulations and capture per-ship state snapshots.

Each ship's ShipState is recorded at every personality tick (~4 Hz) using a
LoggingPersonality wrapper.  The DefaultPersonality runs normally so the
simulation behaves as expected; the logger captures both the state vector and
the weights DefaultPersonality chose (free weak labels useful for comparison).

Output per scenario under data/unlabeled/<scenario-slug>/:
    states.npz    — compressed numpy arrays for ML training
    states.jsonl  — human-readable JSON Lines for ChatGPT labeling

states.npz keys:
    states           float32 [N, 15]  — state_to_vector output per ship-tick
    default_weights  float32 [N,  8]  — DefaultPersonality weights (MODULE_NAMES order)

states.jsonl rows (one JSON object per line):
    {"state": [15 floats], "default_weights": {"ObjectiveModule": float, ...}}

Usage:
    # single scenario
    python collect_data.py --scenario wargames_2d/scenarios/generated/siege-of-ember-gate/scenario.json

    # all scenarios under a directory
    python collect_data.py --scenarios wargames_2d/scenarios/generated/

    # custom output directory
    python collect_data.py --scenarios wargames_2d/scenarios/generated/ --output data/unlabeled/
"""

import argparse
import datetime
import json
import re
import sys
from pathlib import Path

import numpy as np

# ── wargames_2d is not an installed package — inject its path ────────────────
_WG_DIR = Path(__file__).parent / "wargames_2d"
sys.path.insert(0, str(_WG_DIR))

import pygame
pygame.init()  # required for pygame.math.Vector2 used throughout the simulation

from registry import ShipRegistry                                     # noqa: E402
from scenario import load_scenario                                    # noqa: E402
from simulation import setup_scenario                                 # noqa: E402
from behavior.personality import DefaultPersonality, MODULE_NAMES     # noqa: E402
from behavior.personality.logging_personality import LoggingPersonality  # noqa: E402

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "unlabeled"
DEFAULT_MAX_STEPS  = 18_000   # 5 min at 60 fps — applied when scenario has no cap


# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    s = name.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-") or "scenario"


def _unique_dir(base: Path, slug: str) -> Path:
    target = base / slug
    if not target.exists():
        return target
    counter = 2
    while True:
        candidate = base / f"{slug}_{counter}"
        if not candidate.exists():
            return candidate
        counter += 1


# ── Core pipeline ─────────────────────────────────────────────────────────────

def run_and_collect(scenario_path: Path, output_dir: Path) -> int:
    """
    Load one scenario, run it headlessly with a LoggingPersonality injected
    into every ship, then save the collected state snapshots.

    Returns the number of snapshots captured.
    """
    config = load_scenario(str(scenario_path))

    # Force headless mode and apply a step cap if the scenario has none
    config.simulation.headless = True
    if config.simulation.max_steps is None:
        config.simulation.max_steps = DEFAULT_MAX_STEPS

    # Set up world (spawns ships with their scenario personalities).
    # We do NOT call run_scenario() because it would call setup_scenario()
    # internally (clearing the registry and discarding our hot-swapped loggers).
    # Instead we replicate the simulation loop here after a single setup call.
    state = setup_scenario(config)
    initial_counts = state['initial_counts']

    # Hot-swap ALL ships to a single shared LoggingPersonality so every
    # ship's state is captured regardless of faction personality setting.
    logger = LoggingPersonality(DefaultPersonality())
    for ship in ShipRegistry.ships:
        ship.brain.personality = logger

    # ── Simulation loop (mirrors run_scenario in simulation.py) ──────────────
    sim = config.simulation
    dt  = 1.0 / 60.0
    step = 0

    while True:
        alive_factions = [
            fid for fid in initial_counts
            if ShipRegistry.ships_by_faction.get(fid)
        ]
        if len(alive_factions) <= 1:
            break
        if sim.max_steps is not None and step >= sim.max_steps:
            break

        snapshot = list(ShipRegistry.ships)
        for ship in snapshot:
            if ship.is_alive() and ship.brain:
                ship.brain.update(ship, dt, ShipRegistry)
        for ship in snapshot:
            if ship.is_alive() and ship.weapon:
                ship.weapon.update(ship, dt, ShipRegistry)
        for ship in ShipRegistry.ships:
            ship.update_target(ShipRegistry)
        for ship in snapshot:
            if ship.is_alive():
                ship.update(dt)
        step += 1

    # ── Collect outcome ───────────────────────────────────────────────────────
    final_counts  = {fid: len(ShipRegistry.ships_by_faction.get(fid, []))
                     for fid in initial_counts}
    alive_factions = [fid for fid, n in final_counts.items() if n > 0]
    winner = alive_factions[0] if len(alive_factions) == 1 else None

    states, weights = logger.to_arrays()
    n = len(logger.records)

    if n == 0:
        print(f"  WARNING: 0 snapshots captured for {config.name}", file=sys.stderr)
        return 0

    slug = _slugify(config.name)
    out_dir = _unique_dir(output_dir, slug)
    save_outputs(states, weights, config.name, out_dir, config)

    outcome_str = f"winner=faction{winner}" if winner is not None else "draw"
    print(f"  {config.name}: {n:,} snapshots, {step} steps, "
          f"{outcome_str} → {out_dir}")

    return n


def save_outputs(
    states: np.ndarray,
    weights: np.ndarray,
    scenario_name: str,
    out_dir: Path,
    config=None,
) -> None:
    """Write states.npz, states.jsonl, and meta.json to out_dir (created if absent)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # NPZ — compact, fast to load for training
    np.savez_compressed(
        out_dir / "states.npz",
        states=states,
        default_weights=weights,
    )

    # JSONL — human-readable, suitable for ChatGPT labeling prompts
    with open(out_dir / "states.jsonl", "w", encoding="utf-8") as f:
        for state_row, weight_row in zip(states, weights):
            record = {
                "state": [round(float(v), 6) for v in state_row],
                "default_weights": {
                    m: round(float(w), 6)
                    for m, w in zip(MODULE_NAMES, weight_row)
                },
            }
            f.write(json.dumps(record) + "\n")

    # meta.json — per-faction ship/weapon stats for the labeling prompt
    if config is not None:
        meta = {
            "scenario_name": scenario_name,
            "factions": {
                str(fc.id): {
                    "ship_count":      fc.ship_count,
                    "hp":              fc.ship.hp,
                    "shield":          fc.ship.shield,
                    "shield_regen":    fc.ship.shield_regen,
                    "max_speed":       fc.ship.max_speed,
                    "weapon_range":    fc.weapon.range,
                    "damage":          fc.weapon.damage,
                    "cassette_size":   fc.weapon.cassette_size,
                    "reload_duration": fc.weapon.reload_duration,
                }
                for fc in config.factions
            },
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture per-ship state snapshots from headless scenario runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--scenario",
        type=Path,
        metavar="PATH",
        help="Path to a single scenario.json file",
    )
    group.add_argument(
        "--scenarios",
        type=Path,
        metavar="DIR",
        help="Directory — collects data from every */scenario.json found inside",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Resolve list of scenario paths
    if args.scenario:
        scenario_paths = [args.scenario]
    else:
        scenario_paths = sorted(args.scenarios.glob("*/scenario.json"))
        if not scenario_paths:
            print(f"No scenario.json files found under {args.scenarios}", file=sys.stderr)
            sys.exit(1)

    print(f"Collecting data from {len(scenario_paths)} scenario(s) → {args.output}\n")

    total_snapshots = 0
    successes = 0
    for i, path in enumerate(scenario_paths, 1):
        print(f"[{i}/{len(scenario_paths)}] {path}")
        try:
            n = run_and_collect(path, args.output)
            total_snapshots += n
            successes += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
        print()

    print(f"Done. {successes}/{len(scenario_paths)} scenario(s) completed, "
          f"{total_snapshots:,} total snapshots saved.")
    if successes < len(scenario_paths):
        sys.exit(1)


if __name__ == "__main__":
    main()
