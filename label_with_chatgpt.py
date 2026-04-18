#!/usr/bin/env python3
"""
label_with_chatgpt.py — Send unlabeled ship-state snapshots to ChatGPT for tactical weight labeling.

Each state is described in human-readable form (whole percentages, pixel distances,
ship stat context) and sent to ChatGPT as a one-shot tactical reasoning request.
The model outputs a reasoning paragraph and 8 module weights in [0, 1].

Output per scenario under data/labeled/<scenario-slug>/:
    labels.jsonl  — one JSON object per labeled state:
                    {"state": [15 floats], "reasoning": "...", "weights": {...8 floats...}}

Usage:
    # single scenario directory
    python label_with_chatgpt.py --unlabeled-scenario data/unlabeled/symmetric-standoff-twin-lances-at-midfield

    # all scenarios under a directory (every 15th row)
    python label_with_chatgpt.py --unlabeled-dir data/unlabeled --sample 15

    # custom output directory and model
    python label_with_chatgpt.py --unlabeled-dir data/unlabeled --output data/labeled --model gpt-5.4-mini
"""

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Tactical modules only — BoundsModule and CollisionAvoidanceModule are
#    static weights set by the game engine and excluded from labeling. ─────────
MODULE_NAMES = [
    'ObjectiveModule', 'FlockingModule', 'EvasionModule', 'AttackModule',
]

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "data" / "labeled"
DEFAULT_MODEL      = "gpt-5.4-mini"
DEFAULT_SAMPLE     = 1      # process every row; use --sample 15 for one per personality tick
DEFAULT_WORKERS    = 10     # concurrent API calls


# ── System prompt ─────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    return """\
You are a tactical AI advisor for autonomous starship combat.
You analyze per-ship battle state snapshots and assign weights to 6 behavioral modules
that blend into the ship's final movement vector.  Each weight is a float in [0.0, 1.0].

A weight of 0 means the module has no influence; 1.0 means full influence.
The modules are summed into one steering vector — higher weights dominate movement.
Having multiple high-weight modules that pull in opposite directions produces a weak,
indecisive vector; one module should clearly dominate in each situation.

═══════════════════════════════════════════════
MODULE REFERENCE
═══════════════════════════════════════════════
ObjectiveModule          Navigate toward the team's designated objective zone — a
                         geographic area to capture and hold on the map.  This is NOT
                         the current combat target; the objective zone may be far from
                         any enemy.  Ships should drift toward it when not otherwise
                         occupied, and actively hold it when inside.
FlockingModule           Unified flocking drive: separation (personal space), alignment
                         (match ally headings), and cohesion (stay near the formation
                         center) all scale together.
                         CAUTION: if this ship is isolated from its fleet, a high
                         FlockingModule weight will drive it back toward allies no matter
                         where they are, even mid attack-run or behind enemy lines.
                         Lower it when the ship needs to break away for an attack run or
                         aggressive evasion.
EvasionModule            Flee from the nearest enemy threat. Use when reloading or taking
                         fire. When the shield is fully depleted, max this out — the ship
                         will fall back under friendly cover (FlockingModule above 0 helps).
                         Shields recharge, so a shielded ship can afford to hold position;
                         an unshielded or damaged ship cannot.
AttackModule             Two distinct modes depending on "Weapon on target":

                         ► CLOSING mode  (Weapon on target: NO, weapon loaded)
                           Ship physically moves toward the target to close range
                           and establish a firing position.  Set HIGH to engage
                           aggressively; set LOW to hold distance or disengage.

                         ► LOCK mode  (Weapon on target: YES)
                           Ship is already in a firing position.  AttackModule now
                           locks the ship's facing onto the target and suppresses
                           forward charging — the ship holds its range and fires.
                           Lateral and evasive movement remains fully active.
                           Set HIGH to stay locked on target; set LOW to disengage.


═══════════════════════════════════════════════
STAT REFERENCE  (game-calibrated ranges)
═══════════════════════════════════════════════
HP:              10 = critically frail  |  60 = lightly armored  |  100 = standard
                 200 = heavily armored  |  265 = maximum armor
Shield:           0 = unshielded       |  100 = light shield     |  200 = standard
                 400 = heavy shield    (absorbs damage before HP)
Shield regen:     0 = none             |  8/s = slow regeneration |  20/s = fast
Max speed:       75 px/s = sluggish    |  100 = standard         |  130 = agile
Weapon range:   220 px = close-quarters|  400 = standard         |  610 = long-range sniper
Damage/shot:      8 = light            |  10 = standard          |  20 = heavy  |  38 = devastating
Cassette size:    2 shots = tiny       |  5 = standard           |  8 = large burst, requires long stay on target
Reload time:      3 s = fast           |  6 s = standard         |  11 s = slow, feel free to move back behind allies


═══════════════════════════════════════════════
TACTICAL GUIDELINES
═══════════════════════════════════════════════
1.  When RELOADING — two cases:
    A. (Almost) Full shield: the shield acts as a buffer — You can afford to take 
       a few hits for the team if the ship is tanky (has high total shield),
       you maintain moderate pressure by setting moderate AttackModule.  
       FlockingModule=0.8, ObjectiveModule=0.6, AttackModule=0.3, EvasionModule=0.3.
    B. Otherwise (damaged or no shield): the ship is exposed — retreat behind allies.
       EvasionModule=0.8–1.0 (higher urgency = higher), AttackModule=0.0,
       FlockingModule=0.1–0.3.  Long reload times warrant maximum evasion.

3.  When weapon is loaded and Weapon on target: NO (closing mode) — AttackModule=1.0
    to charge in and establish firing position.  Lower FlockingModule (0.1–0.3) to
    unlock the ship from formation heading.  Lower EvasionModule (0.1) to avoid
    breaking the attack run.

4.  When Weapon on target: YES (lock mode) — AttackModule=1.0 to hold the lock and
    fire.  The ship will NOT charge further in.  Suppress other modules to avoid
    breaking the lock.

5.  When NO target is acquired — ObjectiveModule should DOMINATE (0.7–0.9).
    Keep AttackModule at 0.2–0.3 only — it provides a gentle drift toward enemy
    territory.  Do NOT raise AttackModule above 0.4 with no target; without a lock
    it has no combat value and wastes the steering vector.
    FlockingModule=0.5 to hold formation while searching.

6.  FlockingModule vs EvasionModule tension: if near allies (friend_count high relative
    to enemy_count), FlockingModule can be raised safely.

7.  When outnumbered AND near allies: raise FlockingModule for tight formation.
    When outnumbered AND isolated: raise EvasionModule instead.

8.  When target HP < 20%: press the attack — the kill is worth the risk.
    AttackModule +0.2, EvasionModule −0.1.

9.  Inside the objective zone, ObjectiveModule will keep the ship circling to hold it.
    Raise it to hold the zone, lower it to break away for attack/evasion.

10. HP < 30% is a FULL SURVIVAL OVERRIDE — no exceptions:
    EvasionModule=1.0, AttackModule=0.0 (MANDATORY — do not attack while critically
    injured), ObjectiveModule=0.0, FlockingModule=0.1 if allies present else 0.0.

11. One module must clearly dominate (near 1.0).  Weights clustered in the 0.3–0.5
    range produce indecisive behavior.


═══════════════════════════════════════════════
ILLUSTRATIVE EXAMPLES
═══════════════════════════════════════════════
Representative weight sets for common situations.  Adapt to each state — do not copy blindly.

RELOADING, HP 100%, Shield 100% (shielded & healthy — hold formation, maintain pressure and formation):
  FlockingModule=0.8, ObjectiveModule=0.6, AttackModule=0.3, EvasionModule=0.3

RELOADING, HP 80%, Shield 0% (exposed — retreat):
  EvasionModule=0.9, ObjectiveModule=0.4, FlockingModule=0.2, AttackModule=0.0

RELOADING, HP 80%, Shield 0%, long reload ≥7 s (exposed, slow to recover — maximum retreat):
  EvasionModule=1.0, ObjectiveModule=0.4, FlockingModule=0.2, AttackModule=0.0

RELOADING, HP 40%, Shield 0% enemies present (damaged — survival override):
  EvasionModule=1.0, ObjectiveModule=0.0, FlockingModule=0.1, AttackModule=0.0

RELOADING, HP 40%, Shield 100% enemies present (damaged but fully shielded - keep formation):
  EvasionModule=0.5, ObjectiveModule=0.4, FlockingModule=1.0, AttackModule=0.0

NO TARGET acquired, loaded (seek — navigate to objective, drift toward enemies):
  ObjectiveModule=0.8, FlockingModule=0.5, AttackModule=0.3, EvasionModule=0.1

Loaded, Weapon on target: NO (closing mode), HP 90%, Shield = 100% even numbers:
  AttackModule=1.0, EvasionModule=0.1, FlockingModule=0.3, ObjectiveModule=0.4

Loaded, Weapon on target: YES (lock mode), HP 100%, Shield = 100% even numbers:
  AttackModule=1.0, EvasionModule=0.1, ObjectiveModule=0.1, FlockingModule=0.2

HP < 30%, shield ~0% any weapon state (critical — survival override, no exceptions):
  EvasionModule=1.0, AttackModule=0.0, ObjectiveModule=0.0, FlockingModule=0.1
  
Loaded, HP < 30%, unshielded ship (max Shield = 0), - try to carefully engage targets of opportunity getting in objective range:
EvasionModule=0.5, ObjectiveModule=0.6, FlockingModule=0.5, AttackModule=7.0

Inside objective, loaded, Weapon on target: YES, even numbers:
  AttackModule=1.0, ObjectiveModule=0.6, EvasionModule=0.1, FlockingModule=0.2

Loaded, Weapon on target: NO, outnumbered AND isolated (closing but cautious):
  AttackModule=0.6, EvasionModule=0.4, FlockingModule=0.1, ObjectiveModule=0.4


═══════════════════════════════════════════════
OUTPUT FORMAT
═══════════════════════════════════════════════
Respond with raw JSON only — no markdown, no code fences, no extra keys.
Output exactly these 4 modules — do not include BoundsModule or CollisionAvoidanceModule.
{
  "reasoning": "2–3 sentence tactical assessment of this specific situation",
  "weights": {
    "ObjectiveModule": 0.0,
    "FlockingModule": 0.0,
    "EvasionModule": 0.0,
    "AttackModule": 0.0
  }
}
"""


# ── State description ─────────────────────────────────────────────────────────

def _norm_dist_to_px(v: float) -> Optional[float]:
    """Inverse of norm_dist = 1/(1 + d/400).  Returns None when v == 0 (no target)."""
    if v <= 0.0:
        return None
    return (1.0 / v - 1.0) * 400.0


def _tanh_to_speed(v: float) -> float:
    """Inverse of tanh(speed/100).  Clamps to ±0.9999 to avoid atanh(±1)."""
    v = max(-0.9999, min(0.9999, v))
    return math.atanh(v) * 100.0


def _norm_count(v: float) -> str:
    """Inverse of min(n/30, 1.0).  Returns a readable count string."""
    if v >= 1.0:
        return "≥30"
    return str(round(v * 30.0))


def describe_state(state: list[float], meta: Optional[dict]) -> str:
    """
    Convert a 20-float normalized state vector to a human-readable prompt block.

    State vector order (matches state_to_vector in behavior/personality/__init__.py):
    [0]  hp_frac
    [1]  shield_frac
    [2]  weapon_loaded  (float 0/1)
    [3]  ammo_frac
    [4]  reload_progress
    [5]  speed_frac
    [6]  has_target  (float 0/1)
    [7]  norm_dist(target_distance)
    [8]  target_hp_frac
    [9]  tanh(closing_speed / 100)
    [10] weapon_on_target  (float 0/1)
    [11] min(friend_count / 30, 1.0)
    [12] min(enemy_count  / 30, 1.0)
    [13] norm_dist(objective_distance)
    [14] inside_objective  (float 0/1)
    [15] max_hp / 265.0
    [16] max_speed / 130.0
    [17] weapon_range / 610.0
    [18] weapon_damage / 38.0
    [19] reload_duration / 11.0
    """
    (hp_frac, shield_frac, weapon_loaded_f, ammo_frac, reload_progress,
     speed_frac, has_target_f, norm_target_dist, target_hp_frac,
     closing_speed_tanh, can_fire_f, friend_norm, enemy_norm,
     norm_obj_dist, inside_obj_f, *profile) = state

    weapon_loaded = weapon_loaded_f > 0.5
    has_target    = has_target_f    > 0.5
    can_fire      = can_fire_f      > 0.5
    inside_obj    = inside_obj_f    > 0.5

    hp_pct     = round(hp_frac     * 100)
    shield_pct = round(shield_frac * 100)
    ammo_pct   = round(ammo_frac   * 100)
    reload_pct = round(reload_progress * 100)
    speed_pct  = round(speed_frac  * 100)

    friend_str = _norm_count(friend_norm)
    enemy_str  = _norm_count(enemy_norm)

    # Per-ship profile decoded from vector indices 15-19
    if len(profile) >= 5:
        ship_max_hp    = round(profile[0] * 265.0)
        ship_max_speed = round(profile[1] * 130.0)
        weapon_range   = profile[2] * 610.0
        weapon_damage  = round(profile[3] * 38.0)
        reload_dur     = profile[4] * 11.0
    else:
        ship_max_hp = ship_max_speed = weapon_damage = None
        reload_dur  = None
        weapon_range = 400.0   # fallback

    # Faction overview from meta (ship counts, faction comparison)
    factions_str = ""
    if meta:
        faction_lines = []
        for fid, fc in meta.get("factions", {}).items():
            hp    = fc.get("hp",    100)
            sh    = fc.get("shield", 200)
            regen = fc.get("shield_regen", 0)
            spd   = fc.get("max_speed", 100)
            regen_str = f"{regen:.0f}/s regen" if regen > 0 else "no regen"
            faction_lines.append(
                f"  Faction {fid}: {fc.get('ship_count','?')} ships | "
                f"{hp:.0f} HP | {sh:.0f} shield ({regen_str}) | {spd:.0f} px/s max"
            )
        if faction_lines:
            factions_str = "\nFaction overview:\n" + "\n".join(faction_lines)

    # ── Ship status block ─────────────────────────────────────────────────────
    lines = ["Ship status:"]

    # HP — include max HP for absolute context
    if ship_max_hp is not None:
        lines.append(f"  Health:   {hp_pct}% HP  (max {ship_max_hp} HP)")
    else:
        lines.append(f"  Health:   {hp_pct}% HP")

    # Shield
    if shield_frac > 0.0:
        lines.append(f"  Shield:   {shield_pct}% charged")
    else:
        lines.append(f"  Shield:   0% (no shield or depleted)")

    # Weapon — include profile inline so this ship's stats are unambiguous
    profile_suffix = ""
    if weapon_damage is not None:
        profile_suffix = (
            f"  |  range {weapon_range:.0f} px"
            f"  |  {weapon_damage} dmg/shot"
            f"  |  {reload_dur:.1f} s reload"
        )
    if weapon_loaded:
        if ammo_pct < 100:
            lines.append(f"  Weapon:   LOADED — {ammo_pct}% ammo remaining{profile_suffix}")
        else:
            lines.append(f"  Weapon:   LOADED — full ammo{profile_suffix}")
    else:
        lines.append(f"  Weapon:   RELOADING — {reload_pct}% complete{profile_suffix}")

    # Speed — include max speed for context
    if ship_max_speed is not None:
        lines.append(f"  Speed:    {speed_pct}% of max  (max {ship_max_speed} px/s)")
    else:
        lines.append(f"  Speed:    {speed_pct}% of max")

    # ── Target block ──────────────────────────────────────────────────────────
    lines.append("")
    lines.append("Target:")
    if not has_target:
        lines.append("  No target acquired")
    else:
        target_dist_px = _norm_dist_to_px(norm_target_dist)
        target_hp_pct  = round(target_hp_frac * 100)
        closing_speed  = _tanh_to_speed(closing_speed_tanh)

        if target_dist_px is not None:
            in_range = target_dist_px <= weapon_range
            range_tag = "IN RANGE" if in_range else f"OUT OF RANGE (weapon: {weapon_range:.0f} px)"
            lines.append(f"  Distance: {target_dist_px:.0f} px  ({range_tag})")
        else:
            lines.append(f"  Distance: unknown")

        lines.append(f"  Target HP: {target_hp_pct}% remaining")

        if closing_speed > 1.0:
            lines.append(f"  Closing speed: +{closing_speed:.0f} px/s (approaching)")
        elif closing_speed < -1.0:
            lines.append(f"  Closing speed: {closing_speed:.0f} px/s (retreating)")
        else:
            lines.append(f"  Closing speed: ~0 px/s (lateral or neutral)")

        if can_fire:
            lines.append(f"  Weapon on target: YES  (loaded, aimed, in range, clear LoS — lock mode active)")
        elif weapon_loaded:
            lines.append(f"  Weapon on target: NO   (loaded but not yet aimed or in range — raise AttackModule to close)")
        else:
            lines.append(f"  Weapon on target: NO   (RELOADING — AttackModule must be 0)")

    # ── Situation block ───────────────────────────────────────────────────────
    lines.append("")
    lines.append("Situation:")

    fi = int(friend_str) if friend_str != "≥30" else 30
    ei = int(enemy_str)  if enemy_str  != "≥30" else 30
    if fi > ei:
        balance = "(we have the numbers advantage)"
    elif fi < ei:
        balance = "(we are outnumbered)"
    else:
        balance = "(even numbers)"
    lines.append(f"  Allies: {friend_str} remaining | Enemies: {enemy_str} remaining  {balance}")

    obj_dist_px = _norm_dist_to_px(norm_obj_dist)
    if obj_dist_px is None:
        lines.append("  Objective: no objective assigned")
    elif inside_obj:
        lines.append(f"  Objective: INSIDE zone (holding)")
    else:
        lines.append(f"  Objective: ~{obj_dist_px:.0f} px away | NOT inside zone")

    # ── Faction overview (from meta) ──────────────────────────────────────────
    if factions_str:
        lines.append(factions_str)

    return "\n".join(lines)


# ── ChatGPT call ──────────────────────────────────────────────────────────────

def call_chatgpt(
    desc: str,
    system: str,
    client: OpenAI,
    model: str,
) -> dict:
    """
    Send one state description to ChatGPT and return {"reasoning": str, "weights": dict}.
    Retries once on malformed JSON or missing keys.
    """
    def _do_call() -> dict:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",  "content": system},
                {"role": "user",    "content": desc},
            ],
            temperature=0.3,  # low variance — we want consistent tactical reasoning
        )
        raw = response.choices[0].message.content.strip()
        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw)
        return parsed

    for attempt in range(2):
        try:
            parsed = _do_call()
            # Validate
            if "reasoning" not in parsed or "weights" not in parsed:
                raise ValueError("Response missing 'reasoning' or 'weights' key")
            w = parsed["weights"]
            missing = [m for m in MODULE_NAMES if m not in w]
            if missing:
                raise ValueError(f"weights missing modules: {missing}")
            # Clamp all weights to [0, 1]
            parsed["weights"] = {m: max(0.0, min(1.0, float(w[m]))) for m in MODULE_NAMES}
            return parsed
        except (json.JSONDecodeError, ValueError, KeyError) as exc:
            if attempt == 0:
                print(f"    [retry] malformed response: {exc}", file=sys.stderr)
            else:
                raise ValueError(f"ChatGPT returned invalid response after retry: {exc}") from exc


# ── Stratified sampling ───────────────────────────────────────────────────────

def _assign_stratum(state: list) -> tuple:
    """
    Map a 15-float state vector to a stratum key.
    Six tactically significant dimensions → up to 96 theoretical bins.

    State indices used:
      [0]  hp_frac        [2]  weapon_loaded   [6]  has_target
      [10] can_fire       [11] friend_norm      [12] enemy_norm     [14] inside_obj
    """
    hp_bin      = 0 if state[0] < 0.30 else (1 if state[0] < 0.70 else 2)
    outnumbered = state[12] > state[11] + 0.067   # enemy_norm > friend_norm by ~2 ships
    return (
        int(state[2]  > 0.5),                          # weapon_loaded
        hp_bin,                                         # 0=low / 1=mid / 2=high
        int(state[6]  > 0.5),                          # has_target
        int(state[10] > 0.5 and state[6] > 0.5),      # can_fire (only when has_target)
        int(outnumbered),                               # enemy surplus
        int(state[14] > 0.5),                          # inside_objective
    )


def _stratified_sample(all_lines: list, n: int) -> list:
    """
    Return n row indices sampled proportionally across strata, minimum 1 per
    non-empty stratum.  Uses the Largest Remainder Method for exact allocation.
    """
    strata: dict = defaultdict(list)
    for i, line in enumerate(all_lines):
        line = line.strip()
        if not line:
            continue
        strata[_assign_stratum(json.loads(line)["state"])].append(i)

    total = sum(len(v) for v in strata.values())
    n = min(n, total)

    # Proportional float budget, guaranteed ≥ 1.0 per stratum, re-scaled to n
    budgets = {k: max(1.0, n * len(v) / total) for k, v in strata.items()}
    scale   = n / sum(budgets.values())
    budgets = {k: v * scale for k, v in budgets.items()}

    # Largest Remainder Method → integers summing to exactly n
    floors   = {k: int(v) for k, v in budgets.items()}
    leftover = n - sum(floors.values())
    for k in sorted(budgets, key=lambda k: -(budgets[k] % 1))[:leftover]:
        floors[k] += 1

    chosen = []
    for k, indices in strata.items():
        take = min(floors[k], len(indices))
        if take > 0:
            chosen.extend(random.sample(indices, take))
    if len(chosen) > n:        # edge case: stratum caps caused overshoot
        chosen = random.sample(chosen, n)
    return sorted(chosen)


def _print_stratum_report(all_lines: list, chosen_indices: list, scenario_name: str) -> None:
    """Print a compact table showing how many rows were drawn from each stratum."""
    chosen_set = set(chosen_indices)
    counts: dict = defaultdict(int)
    for i in chosen_indices:
        line = all_lines[i].strip()
        if line:
            counts[_assign_stratum(json.loads(line)["state"])] += 1

    # Count total rows per stratum for context
    total_per: dict = defaultdict(int)
    for i, line in enumerate(all_lines):
        line = line.strip()
        if line:
            total_per[_assign_stratum(json.loads(line)["state"])] += 1

    labels = ("loaded", "hp", "target", "canfire", "outnmbrd", "inzone")
    print(f"  Strata for {scenario_name} ({len(counts)} non-empty):")
    header = f"    {'loaded':>6} {'hp':>4} {'tgt':>4} {'fire':>5} {'outnum':>6} {'inzone':>6}  {'drawn':>5} / total"
    print(header)
    for k in sorted(counts, key=lambda k: -counts[k]):
        drawn = counts[k]
        total = total_per[k]
        print(f"    {k[0]:>6} {k[1]:>4} {k[2]:>4} {k[3]:>5} {k[4]:>6} {k[5]:>6}  {drawn:>5} / {total:>6,}")


# ── Scenario labeler ──────────────────────────────────────────────────────────

def label_scenario(
    unlabeled_dir: Path,
    output_dir: Path,
    client: OpenAI,
    model: str,
    sample: int,
    workers: int = DEFAULT_WORKERS,
    random_n: Optional[int] = None,
    stratified_n: Optional[int] = None,
    append: bool = False,
) -> int:
    """
    Label rows from unlabeled_dir/states.jsonl.

    Sampling modes:
      stratified_n set — stratified proportional sample (no resume; overwrites output)
      random_n set     — pick random_n rows at random (no resume; overwrites output)
      sample N         — label every Nth row in order (supports resume)

    append=True — use write mode "a" for stratified/random so existing rows are
    preserved and new labels are appended alongside them.

    Returns number of newly labeled rows.
    """
    states_file = unlabeled_dir / "states.jsonl"
    if not states_file.exists():
        print(f"  SKIP: no states.jsonl in {unlabeled_dir}", file=sys.stderr)
        return 0

    # Load meta.json for stat context (optional)
    meta_file = unlabeled_dir / "meta.json"
    meta = None
    if meta_file.exists():
        with open(meta_file, encoding="utf-8") as f:
            meta = json.load(f)
        scenario_name = meta.get("scenario_name", unlabeled_dir.name)
    else:
        scenario_name = unlabeled_dir.name

    out_dir = output_dir / unlabeled_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_file = out_dir / "labels.jsonl"

    # Load all rows once
    with open(states_file, encoding="utf-8") as f:
        all_lines = f.readlines()
    total_rows = len(all_lines)

    if stratified_n is not None:
        # ── Stratified mode — proportional sample across state-space bins ─────
        chosen_indices = _stratified_sample(all_lines, stratified_n)
        rows_to_label  = len(chosen_indices)
        write_mode     = "a" if append else "w"
        already_done   = 0
        _print_stratum_report(all_lines, chosen_indices, scenario_name)
        print(f"  {scenario_name}: {total_rows:,} total rows, "
              f"stratified={stratified_n}, labeling {rows_to_label} rows")
    elif random_n is not None:
        # ── Random mode — pick N rows uniformly at random, no resume ─────────
        n = min(random_n, total_rows)
        chosen_indices = sorted(random.sample(range(total_rows), n))
        rows_to_label  = n
        write_mode     = "a" if append else "w"
        already_done   = 0
        print(f"  {scenario_name}: {total_rows:,} total rows, "
              f"random={n}, labeling {rows_to_label} rows")
    else:
        # ── Sequential mode — every Nth row, with resume ──────────────────────
        chosen_indices = [i for i in range(total_rows) if i % sample == 0]
        # Resume: count already-written rows
        already_done = 0
        if labels_file.exists():
            with open(labels_file, encoding="utf-8") as f:
                already_done = sum(1 for _ in f)
        chosen_indices = chosen_indices[already_done:]
        rows_to_label  = len(chosen_indices)
        write_mode     = "a"
        if rows_to_label == 0:
            print(f"  {scenario_name}: already fully labeled ({already_done} rows)")
            return 0
        print(f"  {scenario_name}: {total_rows:,} total rows, "
              f"sample={sample}, labeling {rows_to_label} rows "
              f"(resuming from {already_done})")

    system = build_system_prompt()
    newly_done = 0

    def _process(row_idx: int):
        row  = json.loads(all_lines[row_idx])
        desc = describe_state(row["state"], meta)
        result = call_chatgpt(desc, system, client, model)
        return row_idx, row["state"], result

    with open(labels_file, write_mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futs = {executor.submit(_process, idx): idx for idx in chosen_indices}
            for future in as_completed(futs):
                try:
                    _, state, result = future.result()
                except Exception as exc:
                    print(f"    [row {futs[future]}] FAILED: {exc}", file=sys.stderr)
                    continue
                out.write(json.dumps({
                    "state":     state,
                    "reasoning": result["reasoning"],
                    "weights":   result["weights"],
                }) + "\n")
                out.flush()
                newly_done += 1
                if newly_done % 10 == 0:
                    print(f"    {newly_done}/{rows_to_label} labeled…")

    print(f"  → {newly_done} new rows written to {labels_file}")
    return newly_done


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label ship-state snapshots with ChatGPT tactical weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--unlabeled-scenario",
        type=Path,
        metavar="DIR",
        help="Single unlabeled scenario directory (contains states.jsonl)",
    )
    group.add_argument(
        "--unlabeled-dir",
        type=Path,
        metavar="DIR",
        help="Parent directory — labels every */states.jsonl found inside",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=DEFAULT_SAMPLE,
        metavar="N",
        help=(
            f"Process every Nth row in order (default: {DEFAULT_SAMPLE}). "
            "Recommend 15 — one per personality tick at 60 fps. "
            "Supports resume. Mutually exclusive with --random."
        ),
    )
    parser.add_argument(
        "--random",
        type=int,
        default=None,
        metavar="N",
        dest="random_n",
        help=(
            "Pick N rows at random from each scenario. "
            "Overwrites existing output (no resume). "
            "Mutually exclusive with --sample and --stratified."
        ),
    )
    parser.add_argument(
        "--stratified",
        type=int,
        default=None,
        metavar="N",
        dest="stratified_n",
        help=(
            "Pick N rows per scenario using stratified sampling across 6 "
            "tactically significant state-space dimensions. "
            "Overwrites existing output (no resume). "
            "Mutually exclusive with --sample and --random."
        ),
    )
    parser.add_argument(
        "--append",
        action="store_true",
        default=False,
        help=(
            "Append to existing labels.jsonl instead of overwriting. "
            "Use with --stratified or --random to top up a previous partial run "
            "without losing already-labeled rows."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Concurrent API workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    exclusive = sum([
        args.random_n is not None,
        args.stratified_n is not None,
        args.sample != DEFAULT_SAMPLE,
    ])
    if exclusive > 1:
        parser.error("--random, --stratified, and --sample are mutually exclusive")

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: no OpenAI API key found. "
            "Set the OPENAI_API_KEY environment variable or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    # Resolve list of unlabeled scenario directories
    if args.unlabeled_scenario:
        scenario_dirs = [args.unlabeled_scenario]
    else:
        scenario_dirs = sorted(
            d for d in args.unlabeled_dir.iterdir()
            if d.is_dir() and (d / "states.jsonl").exists()
        )
        if not scenario_dirs:
            print(f"No states.jsonl files found under {args.unlabeled_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Labeling {len(scenario_dirs)} scenario(s) → {args.output}\n")

    total_labeled = 0
    successes = 0
    for i, sdir in enumerate(scenario_dirs, 1):
        print(f"[{i}/{len(scenario_dirs)}] {sdir}")
        try:
            n = label_scenario(
                unlabeled_dir=sdir,
                output_dir=args.output,
                client=client,
                model=args.model,
                sample=args.sample,
                workers=args.workers,
                random_n=args.random_n,
                stratified_n=args.stratified_n,
                append=args.append,
            )
            total_labeled += n
            successes += 1
        except Exception as exc:
            print(f"  FAILED: {exc}", file=sys.stderr)
        print()

    print(
        f"Done. {successes}/{len(scenario_dirs)} scenario(s) completed, "
        f"{total_labeled:,} total rows labeled."
    )
    if successes < len(scenario_dirs):
        sys.exit(1)


if __name__ == "__main__":
    main()
