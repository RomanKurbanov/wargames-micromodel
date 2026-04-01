#!/usr/bin/env python3
"""
generate_scenarios.py — Generate diverse wargames_2d battle scenarios via ChatGPT.

Each run asks ChatGPT to design one or more tactical battle scenarios for the
autonomous ship simulation.  For each scenario the model emits:
  - 1–3 paragraphs of strategic reasoning
  - A complete scenario JSON matching the wargames_2d schema

Both are saved together under wargames_2d/scenarios/generated/<scenario-slug>/.

Usage:
    python generate_scenarios.py --count 5
    python generate_scenarios.py --count 3 --theme "guerrilla hit-and-run"
    python generate_scenarios.py --count 1 --api-key sk-...
    OPENAI_API_KEY=sk-... python generate_scenarios.py --count 10

Dependencies (install once):
    pip install -r requirements-generator.txt
"""

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Optional

from openai import OpenAI

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "wargames_2d" / "scenarios" / "generated"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_TEMPERATURE = 0.9   # high variance → diverse scenarios
DEFAULT_COUNT = 5

WORLD_HALF_X = 1400.0
WORLD_HALF_Y = 800.0
VALID_PERSONALITIES = {"default", "none"}   # "ml:" requires a trained model not yet available
MIN_SPAWN_SEPARATION = 300.0  # px — factions must not spawn on top of each other

# Archetypes cycled through in order when --theme is not provided
ARCHETYPES = [
    "Symmetric standoff: two equal factions face each other with mirrored stats",
    "Swarm vs sniper: many fast fragile ships vs few slow durable long-range ships",
    "Guerrilla hit-and-run: fast light faction with short reload vs slow heavy faction",
    "Siege: one faction must push into the enemy spawn zone; give them asymmetric objectives",
    "Three-way free-for-all: three factions (ids 0, 1, 2) fight over a central objective",
    "Shield wall vs fragile horde: one side has high shield/regen, the other has raw DPS",
    "Skirmish: small counts (5-10 ships each), close-range weapons, dense obstacles",
    "Attrition standoff: both sides have long reload times and high HP — slow grind",
]


# ── Prompts ──────────────────────────────────────────────────────────────────

def build_system_prompt() -> str:
    """Return the static system prompt sent to ChatGPT on every call."""
    return """\
You are a wargame scenario designer for a 2D autonomous ship battle simulation.
Your task is to produce varied, strategically interesting battle scenarios that will
be used as training data for an AI micro-model.  Each scenario defines two or more
factions of autonomous ships.  Ships use modular steering AI: they pursue objectives,
avoid collisions, engage enemies, and evade threats.  Your scenarios should create
diverse tactical situations so the AI learns a wide range of behaviors.


═══════════════════════════════════════════════
WORLD GEOMETRY
═══════════════════════════════════════════════
- The world is 2800 × 1600 pixels, centered at (0, 0).
- Valid coordinate range: x ∈ [-1400, 1400],  y ∈ [-800, 800].
- world_bounds controls the boundary-repulsion radius; keep it between 1800 and 2200.
- Ships cluster near their spawn_center within spawn_radius pixels.
- SPAWN SAFETY RULE: spawn centers must be at least (weapon_range_A + weapon_range_B)
  pixels apart so factions have time to react.  Minimum absolute separation: 300 px.
  Example: two factions with range 400 each → spawn centers ≥ 800 px apart.


═══════════════════════════════════════════════
SCENARIO ARCHETYPES  (rotate through these — do not repeat the same one twice)
═══════════════════════════════════════════════
1. Symmetric standoff      — two equal factions face each other, mirrored stats.
2. Swarm vs sniper         — many fast fragile ships vs few slow durable long-range ships.
3. Guerrilla hit-and-run   — fast light faction with short reload vs slow heavy faction.
4. Siege                   — one faction pushes into enemy spawn zone; asymmetric objectives.
5. Three-way free-for-all  — three factions (ids 0, 1, 2) compete over a central objective.
6. Shield wall vs horde    — one side has high shield/regen, the other has raw DPS.
7. Skirmish                — small counts (5–10 ships each), close-range, dense obstacles.
8. Attrition standoff      — both sides have long reload, high HP; slow grinding battle.


═══════════════════════════════════════════════
STAT GUIDELINES  (these are ranges, not defaults)
═══════════════════════════════════════════════
SHIP STATS:
  max_speed             50  – 250   px/s     fast swarm ≈ 180, heavy tank ≈ 60
  forward_acceleration  20  – 120   px/s²
  brake_acceleration    80  – 300   px/s²
  turn_rate             60  – 360   deg/s    agile ≈ 280, lumbering ≈ 80
  lateral_damping       1.5 – 6.0
  hp                    25  – 300   HP       fragile ≈ 35, tank ≈ 250
  shield                0   – 500            0 = no shield at all
  shield_regen          0   – 20    pts/s    0 = no regen
  size                  6   – 20    px       swarm ≈ 7, dreadnought ≈ 18
  visual_range          400 – 1200  px

WEAPON STATS:
  range                 150 – 700   px       brawler ≈ 200, sniper ≈ 600
  damage                5   – 50    HP/shot  rapid-fire ≈ 6, sniper ≈ 40
  fire_rate             1   – 15    shots/s  precision ≈ 1.5, rapid-fire ≈ 12
  max_fire_angle        10  – 90    deg      sniper ≈ 12, shotgun ≈ 60
  cassette_size         1   – 20    (INTEGER — never a float like 5.0)
  reload_duration       2   – 15    s

ASYMMETRY RULE:
  Fewer ships must mean meaningfully stronger stats.
  For 10 ships vs 40 ships → the 10 should have roughly 3–4× individual combat power
  (hp × damage × fire_rate / reload_cycle is a useful proxy).

OBSTACLE GUIDELINES:
  count        0  – 25    open field = 0, dense = 20+, ideal variety = 8–18
  min_size    40  – 80    px
  max_size    80  – 150   px
  Use a fixed integer "seed" to make the layout reproducible.
  Setting "enabled": false is acceptable for wide open-field scenarios.


═══════════════════════════════════════════════
BALANCE CONSTRAINTS  (mandatory — check before finalising stats)
═══════════════════════════════════════════════
These rules prevent one-sided battles where one faction is eliminated before any
meaningful engagement occurs.

1. SURVIVABILITY FLOOR — no ship should be one-shot by a single enemy hit.
   RULE: hp + shield  ≥  2 × (highest enemy damage per shot)
   Example: enemy sniper damage = 35 → your swarm ships need hp + shield ≥ 70.
   Give zero-shield swarm ships enough hp, or add a small shield (20–60) to absorb hits.

2. SHIELD REGEN CAP — shield regen must not outpace enemy DPS.
   Define cycle_dps = (cassette_size × damage) / (cassette_size / fire_rate + reload_duration)
   RULE: shield_regen  ≤  (enemy faction total cycle_dps / your ship count) × 0.4
   Intuition: one ship's regen should recover at most 40 % of what one enemy ship deals
   per second on average.  High-regen "shield wall" ships should have shield_regen ≤ 8
   unless the enemy has very high sustained DPS (e.g. 12+ rapid-fire ships).

3. TOTAL POWER BALANCE — the stronger faction must not exceed 2× total combat power.
   Define power = ship_count × (hp + shield) × cycle_dps  for each faction.
   RULE: max(power_A, power_B) / min(power_A, power_B)  ≤  2.0
   Asymmetric archetypes (swarm vs sniper, siege) may reach 1.8–2.0×; symmetric
   archetypes should be near 1.0×.  If your numbers exceed 2.0× after drafting,
   adjust ship counts, hp, or damage until the ratio is within range.

4. SIEGE OBJECTIVES — the defending faction must have an objective that causes them
   to actively contest territory, not just sit at their spawn.
   RULE: for siege scenarios, set the defender's objective to a forward position
   (e.g. midfield or the attacker's spawn) so both sides advance and engage.


═══════════════════════════════════════════════
OUTPUT FORMAT  — follow this exactly
═══════════════════════════════════════════════
Write 1 to 3 paragraphs of strategic reasoning that explain:
  • which archetype this scenario represents
  • why you chose these stats (what tactical dynamic they create)
  • what makes the battle interesting or educational for AI training

Then output the scenario as a JSON code block:

```json
{
  ...
}
```

The JSON MUST be valid — no comments, no trailing commas, no unquoted keys.
The "name" field must be unique and descriptive (e.g. "Swarm vs Sniper — Dense Nebula").
"simulation.headless" must be false (scenarios should be playable interactively).
"personality" for each faction MUST be "default" or "none".


═══════════════════════════════════════════════
COMPLETE JSON SCHEMA  (fill all required fields)
═══════════════════════════════════════════════
Required fields: factions (array, ≥ 2 entries), each faction's id / ship_count / spawn_center.

```json
{
  "name": "...",
  "description": "...",
  "world_bounds": 2000,
  "factions": [
    {
      "id": 0,
      "ship_count": 30,
      "spawn_center": [-400, 0],
      "spawn_radius": 150,
      "objective": {
        "position": [400, 0],
        "patrol_radius": 200
      },
      "ship": {
        "max_speed": 100,
        "forward_acceleration": 40,
        "brake_acceleration": 200,
        "turn_rate": 180,
        "lateral_damping": 3.0,
        "hp": 100,
        "shield": 200,
        "shield_regen": 8,
        "size": 10,
        "visual_range": 800
      },
      "weapon": {
        "range": 400,
        "damage": 10,
        "fire_rate": 5,
        "max_fire_angle": 25,
        "cassette_size": 5,
        "reload_duration": 6
      },
      "personality": "default"
    },
    {
      "id": 1,
      "ship_count": 20,
      "spawn_center": [500, 0],
      "spawn_radius": 120,
      "objective": {
        "position": [-400, 0],
        "patrol_radius": 250
      },
      "ship": {
        "max_speed": 75,
        "forward_acceleration": 25,
        "brake_acceleration": 150,
        "turn_rate": 120,
        "lateral_damping": 2.0,
        "hp": 100,
        "shield": 400,
        "shield_regen": 20,
        "size": 14,
        "visual_range": 1000
      },
      "weapon": {
        "range": 600,
        "damage": 25,
        "fire_rate": 2,
        "max_fire_angle": 15,
        "cassette_size": 3,
        "reload_duration": 10
      },
      "personality": "default"
    }
  ],
  "obstacles": {
    "enabled": true,
    "count": 10,
    "min_size": 50,
    "max_size": 90,
    "world_half_x": 1400,
    "world_half_y": 800,
    "spawn_clearance": 300,
    "seed": 42
  },
  "simulation": {
    "max_steps": 18000,
    "headless": false,
    "output": {
      "file": null,
      "record_interval": 60
    }
  }
}
```
"""


def build_user_prompt(theme: Optional[str], archetype: Optional[str] = None) -> str:
    """Return the per-call user prompt with an explicit archetype assignment."""
    base = (
        "Generate one complete wargames battle scenario following the format "
        "and schema described in the system prompt."
    )
    if theme:
        return f"{base}\n\nUse this archetype: {theme}"
    if archetype:
        return f"{base}\n\nYou MUST use this specific archetype: {archetype}"
    return base


# ── OpenAI call ───────────────────────────────────────────────────────────────

def call_chatgpt(
    client: OpenAI,
    user_prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> str:
    """
    Make one chat completion call and return the raw text response.
    Raises openai.OpenAIError on API failure (caller handles it).
    """
    response = client.chat.completions.create(
        model=model,
        reasoning_effort="medium",
        messages=[
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_json_and_reasoning(response: str) -> tuple[str, str]:
    """
    Parse a ChatGPT response into (reasoning_text, json_str).

    Three extraction strategies, tried in order:
      1. ```json ... ``` labeled fence
      2. ``` ... ``` unlabeled fence whose body starts with {
      3. Brace-balance scan — finds the first { and walks to its matching }

    reasoning_text is everything before the JSON block (stripped).
    Raises ValueError if no JSON-like content is found.
    """
    # Strategy 1 — labeled ```json fence
    m = re.search(r"```json\s*\n(.*?)```", response, re.DOTALL | re.IGNORECASE)
    if m:
        return response[: m.start()].strip(), m.group(1).strip()

    # Strategy 2 — unlabeled fence whose body looks like a JSON object
    m = re.search(r"```\s*\n(\s*\{.*?\})\s*\n```", response, re.DOTALL)
    if m:
        return response[: m.start()].strip(), m.group(1).strip()

    # Strategy 3 — brace-balance scan (handles missing fences)
    start = response.find("{")
    if start == -1:
        raise ValueError("No JSON object found in ChatGPT response")

    depth = 0
    end = -1
    in_string = False
    escape = False
    for i, ch in enumerate(response[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError("Unbalanced braces in ChatGPT response — no valid JSON object found")

    return response[:start].strip(), response[start : end + 1]


# ── Validation ────────────────────────────────────────────────────────────────

def validate_scenario(data: dict) -> list[str]:
    """
    Validate a parsed scenario dict against the wargames_2d schema.
    Returns a list of human-readable error strings.  Empty list means valid.
    """
    errors: list[str] = []

    # ── factions ──────────────────────────────────────────────────────────────
    factions = data.get("factions")
    if not isinstance(factions, list):
        errors.append("'factions' is missing or not a list")
        return errors  # nothing more can be checked without factions
    if len(factions) < 2:
        errors.append(f"Need at least 2 factions, got {len(factions)}")

    seen_ids: set[int] = set()
    spawn_centers: list[tuple[float, float]] = []

    for i, fc in enumerate(factions):
        if not isinstance(fc, dict):
            errors.append(f"factions[{i}]: expected an object, got {type(fc).__name__}")
            continue
        prefix = f"factions[{i}]"

        # id
        if "id" not in fc:
            errors.append(f"{prefix}: missing 'id'")
        else:
            fid = fc["id"]
            if not isinstance(fid, int):
                errors.append(
                    f"{prefix}: 'id' must be int, got {type(fid).__name__} ({fid!r})"
                )
            else:
                if fid in seen_ids:
                    errors.append(f"{prefix}: duplicate faction id {fid}")
                seen_ids.add(fid)

        # ship_count
        if "ship_count" not in fc:
            errors.append(f"{prefix}: missing 'ship_count'")
        else:
            sc = fc["ship_count"]
            if not isinstance(sc, int) or sc <= 0:
                errors.append(
                    f"{prefix}: 'ship_count' must be a positive int, got {sc!r}"
                )

        # spawn_center
        if "spawn_center" not in fc:
            errors.append(f"{prefix}: missing 'spawn_center'")
        else:
            ctr = fc["spawn_center"]
            if (
                not isinstance(ctr, list)
                or len(ctr) != 2
                or not all(isinstance(v, (int, float)) for v in ctr)
            ):
                errors.append(f"{prefix}: 'spawn_center' must be [x, y] numbers")
            else:
                x, y = float(ctr[0]), float(ctr[1])
                if not (-WORLD_HALF_X <= x <= WORLD_HALF_X):
                    errors.append(
                        f"{prefix}: spawn_center x={x} is outside [-{WORLD_HALF_X}, {WORLD_HALF_X}]"
                    )
                if not (-WORLD_HALF_Y <= y <= WORLD_HALF_Y):
                    errors.append(
                        f"{prefix}: spawn_center y={y} is outside [-{WORLD_HALF_Y}, {WORLD_HALF_Y}]"
                    )
                spawn_centers.append((x, y))

        # personality
        personality = fc.get("personality", "default")
        if personality not in VALID_PERSONALITIES:
            errors.append(
                f"{prefix}: personality={personality!r} must be 'default' or 'none' "
                f"(ml: models are not yet available)"
            )

        # ship numeric fields — all must be positive when present
        ship = fc.get("ship", {})
        if isinstance(ship, dict):
            for field in (
                "max_speed", "forward_acceleration", "hp", "size", "visual_range"
            ):
                val = ship.get(field)
                if val is not None and (
                    not isinstance(val, (int, float)) or val <= 0
                ):
                    errors.append(
                        f"{prefix}.ship.{field}: must be a positive number, got {val!r}"
                    )

        # weapon fields
        weapon = fc.get("weapon", {})
        if isinstance(weapon, dict):
            for field in ("range", "damage", "fire_rate", "reload_duration"):
                val = weapon.get(field)
                if val is not None and (
                    not isinstance(val, (int, float)) or val <= 0
                ):
                    errors.append(
                        f"{prefix}.weapon.{field}: must be a positive number, got {val!r}"
                    )
            cassette = weapon.get("cassette_size")
            if cassette is not None and not isinstance(cassette, int):
                errors.append(
                    f"{prefix}.weapon.cassette_size: must be int, "
                    f"got {type(cassette).__name__} ({cassette!r})"
                )

    # ── pairwise spawn separation ─────────────────────────────────────────────
    for a in range(len(spawn_centers)):
        for b in range(a + 1, len(spawn_centers)):
            ax, ay = spawn_centers[a]
            bx, by = spawn_centers[b]
            dist = math.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
            if dist < MIN_SPAWN_SEPARATION:
                errors.append(
                    f"factions[{a}] and factions[{b}] spawn centers are only "
                    f"{dist:.0f} px apart (minimum {MIN_SPAWN_SEPARATION:.0f} px)"
                )


    return errors


# ── File utilities ────────────────────────────────────────────────────────────

def slugify(name: str) -> str:
    """
    Convert a scenario name to a filesystem-safe directory name.
    'Swarm vs Sniper — Dense Nebula' -> 'swarm-vs-sniper-dense-nebula'
    """
    s = name.lower()
    s = re.sub(r"[^\w\s-]", "", s)    # remove non-alphanumeric except hyphens/spaces
    s = re.sub(r"[\s_]+", "-", s)     # spaces and underscores → hyphens
    s = re.sub(r"-+", "-", s)         # collapse multiple hyphens
    s = s.strip("-")
    return s or "scenario"


def save_scenario(
    scenario: dict,
    reasoning: str,
    output_dir: Path,
    slug: str,
) -> Path:
    """
    Write scenario.json and reasoning.txt into output_dir/<slug>/.
    Appends _2, _3, … if the slug directory already exists.
    Returns the Path of the created directory.
    """
    target = output_dir / slug
    if target.exists():
        counter = 2
        while True:
            candidate = output_dir / f"{slug}_{counter}"
            if not candidate.exists():
                target = candidate
                break
            counter += 1

    target.mkdir(parents=True, exist_ok=False)

    (target / "scenario.json").write_text(
        json.dumps(scenario, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (target / "reasoning.txt").write_text(reasoning, encoding="utf-8")

    return target


# ── Generation pipeline ───────────────────────────────────────────────────────

def generate_one(
    client: OpenAI,
    output_dir: Path,
    theme: Optional[str],
    model: str,
    temperature: float,
    archetype: Optional[str] = None,
) -> bool:
    """
    Full pipeline for one scenario: API call → extract → parse → validate → save.
    Returns True on success, False on any failure (errors printed to stderr).

    Note: API rate-limit errors (openai.RateLimitError) will surface as a failure
    here.  For large batches, add a sleep between calls or use --count 1 with a
    shell loop that includes its own backoff.
    """
    user_prompt = build_user_prompt(theme, archetype)

    print("  Calling API...", end="", flush=True)
    try:
        raw = call_chatgpt(client, user_prompt, model, temperature)
    except Exception as exc:
        print(f" FAILED\n  API error: {exc}", file=sys.stderr)
        return False
    print(" done.")

    # Extract JSON and reasoning text
    try:
        reasoning, json_str = extract_json_and_reasoning(raw)
    except ValueError as exc:
        print(f"  Extraction failed: {exc}", file=sys.stderr)
        print(f"  Response preview:\n{raw[:500]}", file=sys.stderr)
        return False

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        print(f"  JSON parse failed: {exc}", file=sys.stderr)
        print(f"  Extracted JSON (first 500 chars):\n{json_str[:500]}", file=sys.stderr)
        return False

    # Validate
    errors = validate_scenario(data)
    if errors:
        print(f"  Validation failed ({len(errors)} error(s)):", file=sys.stderr)
        for err in errors:
            print(f"    - {err}", file=sys.stderr)
        return False

    # Save
    name = data.get("name") or "unnamed"
    slug = slugify(str(name))
    saved_dir = save_scenario(data, reasoning, output_dir, slug)
    print(f"  Saved: {saved_dir}")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate diverse wargames_2d battle scenarios via ChatGPT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of scenarios to generate (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--theme",
        type=str,
        default=None,
        metavar="TEXT",
        help="Optional thematic hint for ChatGPT, e.g. 'guerrilla hit-and-run'",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_TEMPERATURE})",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        metavar="KEY",
        help="OpenAI API key (overrides OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print(
            "Error: no OpenAI API key found.  "
            "Set the OPENAI_API_KEY environment variable or pass --api-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Generating {args.count} scenario(s) → {args.output}")
    print(f"Model: {args.model}   Temperature: {args.temperature}")
    if args.theme:
        print(f"Theme: {args.theme}")
    print()

    successes = 0
    for i in range(args.count):
        archetype = ARCHETYPES[i % len(ARCHETYPES)] if not args.theme else None
        print(f"[{i + 1}/{args.count}] {archetype or args.theme}")
        if generate_one(client, args.output, args.theme, args.model, args.temperature, archetype):
            successes += 1
        print()

    print(f"Done.  {successes}/{args.count} scenario(s) saved successfully.")
    if successes < args.count:
        sys.exit(1)


if __name__ == "__main__":
    main()
