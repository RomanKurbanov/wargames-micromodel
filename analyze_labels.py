#!/usr/bin/env python3
"""
analyze_labels.py — Compare ChatGPT-labeled outputs against TacticalPersonality2.

Loads all data/labeled/*/labels.jsonl, recomputes what TacticalPersonality2 would
output for each state, and reports where the LLM systematically agrees or diverges
from the algorithm — broken down by module and by tactical situation.

Usage:
    python analyze_labels.py
    python analyze_labels.py --data-dir data/labeled/
    python analyze_labels.py --csv report.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np

# ── Module order (must match behavior/personality/__init__.py) ────────────────

# Tactical modules only — BoundsModule and CollisionAvoidanceModule are static
# weights handled by the game engine and excluded from labeling and the model.
MODULE_NAMES = [
    'ObjectiveModule',
    'FlockingModule',
    'EvasionModule',
    'AttackModule',
]
IDX = {m: i for i, m in enumerate(MODULE_NAMES)}
TACTICAL_MODULES = MODULE_NAMES

# ── State vector feature indices ──────────────────────────────────────────────
#   0  hp_frac          4  reload_progress   10 can_fire
#   1  shield_frac      5  speed_frac        11 friend_count (n/30)
#   2  weapon_loaded    6  has_target        12 enemy_count  (n/30)
#   3  ammo_frac        7  target_distance   14 inside_objective
#                       8  target_hp_frac    16 max_speed_stat/130
#                       9  closing_speed     19 reload_duration/11


# ── TacticalPersonality2 reconstruction ──────────────────────────────────────

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _tactical2_row(r) -> list[float]:
    """Return 6-float weight vector for one state row, matching TacticalPersonality2."""
    hp_frac         = r[0]
    shield_frac     = r[1]
    weapon_loaded   = r[2] > 0.5
    ammo_frac       = r[3]
    reload_progress = r[4]
    has_target      = r[6] > 0.5
    target_hp_frac  = r[8]
    can_fire        = r[10] > 0.5
    friend_count    = round(r[11] * 30)
    enemy_count     = round(r[12] * 30)
    inside_obj      = r[14] > 0.5
    fast_ship       = r[16] >= (115.0 / 130.0)
    long_reload     = r[19] >= (7.0  / 11.0)

    outnumbered = enemy_count > friend_count + 1
    isolated    = friend_count <= 1
    dominant    = (friend_count + 1) >= 2 * max(enemy_count, 1)

    objective = evasion = attack = flock = 0.0

    # A: HP critical
    if hp_frac < 0.30:
        out = [0.0] * len(MODULE_NAMES)
        out[IDX['EvasionModule']]  = 1.0
        out[IDX['FlockingModule']] = 0.1 if friend_count > 1 else 0.0
        return out

    # B: Reloading
    if not weapon_loaded:
        urgency = 1.0 - reload_progress
        if shield_frac >= 0.85 and hp_frac >= 0.80:
            objective, flock, evasion, attack = 0.6, 0.8, 0.3, 0.3
        else:
            retreat   = 0.3 if long_reload else 0.0
            evasion   = min(0.7 + 0.2 * urgency + retreat, 1.0)
            attack    = 0.0
            objective = 0.4
            flock     = 0.3 if friend_count > 1 else 0.1
    elif has_target:
        if can_fire:
            # C: Lock mode
            objective = 0.4 if inside_obj else 0.1
            flock, evasion, attack = 0.2, 0.1, 1.0
            if outnumbered:
                evasion = 0.3 if isolated else 0.2
                flock   = 0.1 if isolated else 0.4
        else:
            # D: Closing mode
            objective, flock, evasion, attack = 0.4, 0.3, 0.1, 1.0
            if outnumbered and isolated:
                evasion, attack, flock = 0.4, 0.6, 0.1
            if fast_ship and shield_frac == 0.0 and ammo_frac < 0.30:
                attack, evasion, flock = 0.5, 0.7, 0.1
    else:
        # E: No target
        objective, flock, evasion, attack = 0.8, 0.5, 0.1, 0.3

    # Post-branch modifiers
    if has_target and target_hp_frac < 0.20:
        attack  = min(attack  + 0.2, 1.0)
        evasion = max(evasion - 0.1, 0.0)
    if inside_obj:
        objective = min(objective + 0.2, 1.0)
    if dominant and has_target:
        attack  = min(attack  + 0.1, 1.0)
        evasion = max(evasion - 0.1, 0.0)
    if outnumbered and not isolated:
        flock = min(flock + 0.2, 1.0)
    if shield_frac < 0.15 and hp_frac < 0.75:
        evasion = min(evasion + 0.2, 1.0)

    out = [0.0] * len(MODULE_NAMES)
    out[IDX['ObjectiveModule']] = _clamp01(objective)
    out[IDX['FlockingModule']]  = _clamp01(flock)
    out[IDX['EvasionModule']]   = _clamp01(evasion)
    out[IDX['AttackModule']]    = _clamp01(attack)
    return out


def tactical2_pred(X: np.ndarray) -> np.ndarray:
    return np.array([_tactical2_row(r) for r in X], dtype=np.float32)


# ── Branch classifier ─────────────────────────────────────────────────────────

BRANCHES = ['A', 'B1', 'B2', 'C', 'D', 'E']
BRANCH_LABELS = {
    'A':  'HP critical (hp < 30%)',
    'B1': 'Reloading — shielded & healthy (hold)',
    'B2': 'Reloading — vulnerable (retreat)',
    'C':  'Loaded · has target · can fire (lock)',
    'D':  'Loaded · has target · closing',
    'E':  'Loaded · no target (seek)',
}


def classify_branch(r) -> str:
    hp_frac       = r[0]
    shield_frac   = r[1]
    weapon_loaded = r[2] > 0.5
    has_target    = r[6] > 0.5
    can_fire      = r[10] > 0.5

    if hp_frac < 0.30:
        return 'A'
    if not weapon_loaded:
        return 'B1' if shield_frac >= 0.85 and hp_frac >= 0.80 else 'B2'
    if has_target:
        return 'C' if can_fire else 'D'
    return 'E'


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y_llm, scenario_names)."""
    X_rows, y_rows, names = [], [], []
    files = sorted(data_dir.glob('*/labels.jsonl'))
    if not files:
        raise FileNotFoundError(f"No labels.jsonl files under {data_dir}")
    for path in files:
        slug = path.parent.name
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                X_rows.append(rec['state'])
                y_rows.append([rec['weights'].get(m, 0.0) for m in MODULE_NAMES])
                names.append(slug)
    return (
        np.array(X_rows, dtype=np.float32),
        np.array(y_rows, dtype=np.float32),
        names,
    )


# ── Formatting helpers ────────────────────────────────────────────────────────

def _bar(v: float, width: int = 20, scale: float = 1.0) -> str:
    """ASCII bar proportional to abs(v); prefix + or - to show sign."""
    filled = int(abs(v) / scale * width)
    bar = '█' * min(filled, width)
    sign = '+' if v >= 0 else '−'
    return f"{sign}{bar}"


def _pct(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "—"


SEP  = '─' * 72
SEP2 = '═' * 72


# ── Report sections ───────────────────────────────────────────────────────────

def print_overview(X, y_llm, y_t2) -> None:
    diff = y_llm - y_t2              # positive → LLM higher
    print(SEP2)
    print(f"  {'Module':<28}  {'LLM':>6}  {'T2':>6}  {'Δ mean':>8}  {'|Δ| mean':>8}  bias")
    print(SEP)
    for m in MODULE_NAMES:
        i = IDX[m]
        llm_m  = float(np.mean(y_llm[:, i]))
        t2_m   = float(np.mean(y_t2[:,  i]))
        d_mean = float(np.mean(diff[:,  i]))
        d_abs  = float(np.mean(np.abs(diff[:, i])))
        bar = _bar(d_mean, width=14, scale=0.5)
        print(f"  {m:<28}  {llm_m:>6.3f}  {t2_m:>6.3f}  {d_mean:>+8.3f}  {d_abs:>8.3f}  {bar}")
    print()


def print_branch_breakdown(X, y_llm, y_t2) -> None:
    branches_per_row = [classify_branch(r) for r in X]
    branch_arr = np.array(branches_per_row)
    N = len(X)

    for br in BRANCHES:
        mask = branch_arr == br
        n = int(mask.sum())
        if n == 0:
            continue
        llm_b = y_llm[mask]
        t2_b  = y_t2[mask]
        diff_b = llm_b - t2_b

        print(f"  Branch {br} — {BRANCH_LABELS[br]}")
        print(f"  {n} states ({_pct(n, N)})")
        print(f"  {'Module':<28}  {'LLM':>6}  {'T2':>6}  {'Δ mean':>8}  bias")
        for m in TACTICAL_MODULES:
            i = IDX[m]
            llm_m  = float(np.mean(llm_b[:, i]))
            t2_m   = float(np.mean(t2_b[:,  i]))
            d_mean = float(np.mean(diff_b[:, i]))
            bar = _bar(d_mean, width=12, scale=0.5)
            print(f"  {m:<28}  {llm_m:>6.3f}  {t2_m:>6.3f}  {d_mean:>+8.3f}  {bar}")
        print()


def print_bias_summary(y_llm, y_t2) -> None:
    diff = y_llm - y_t2
    print(f"  {'Module':<28}  {'mean Δ':>8}  {'% rows LLM>T2':>14}  {'% rows LLM<T2':>14}")
    print(SEP)
    for m in TACTICAL_MODULES:
        i = IDX[m]
        d = diff[:, i]
        mean_d = float(np.mean(d))
        pct_hi = 100 * float(np.mean(d >  0.05))
        pct_lo = 100 * float(np.mean(d < -0.05))
        direction = "LLM HIGHER" if mean_d > 0.02 else ("LLM LOWER" if mean_d < -0.02 else "~agree")
        print(f"  {m:<28}  {mean_d:>+8.3f}  {pct_hi:>13.1f}%  {pct_lo:>13.1f}%  {direction}")
    print()


def print_worst_disagreements(X, y_llm, y_t2, top_n: int = 10) -> None:
    """Rows with the highest total absolute divergence across tactical modules."""
    tactical_idx = [IDX[m] for m in TACTICAL_MODULES]
    total_diff = np.sum(np.abs(y_llm[:, tactical_idx] - y_t2[:, tactical_idx]), axis=1)
    top = np.argsort(total_diff)[::-1][:top_n]

    print(f"  {'#':<4}  {'branch':<4}  {'|Δ| total':>10}  {'Obj':>6}  {'Flk':>6}  {'Eva':>6}  {'Atk':>6}")
    print(SEP)
    for rank, idx_r in enumerate(top, 1):
        br   = classify_branch(X[idx_r])
        tot  = total_diff[idx_r]
        diffs = y_llm[idx_r] - y_t2[idx_r]
        cols  = '  '.join(f"{diffs[IDX[m]]:>+6.3f}" for m in TACTICAL_MODULES)
        print(f"  {rank:<4}  {br:<4}  {tot:>10.3f}  {cols}")
    print()


def print_correlation(y_llm, y_t2) -> None:
    print(f"  {'Module':<28}  {'Pearson r':>10}  agreement")
    print(SEP)
    for m in TACTICAL_MODULES:
        i  = IDX[m]
        r  = float(np.corrcoef(y_llm[:, i], y_t2[:, i])[0, 1])
        bar = '█' * int(max(r, 0) * 20)
        print(f"  {m:<28}  {r:>10.3f}  {bar}")
    print()


# ── CSV export ────────────────────────────────────────────────────────────────

def export_csv(X, y_llm, y_t2, path: Path) -> None:
    import csv
    branches = [classify_branch(r) for r in X]
    header = ['branch'] + [f'llm_{m}' for m in MODULE_NAMES] + \
             [f't2_{m}'  for m in MODULE_NAMES] + \
             [f'diff_{m}' for m in MODULE_NAMES]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(header)
        for br, llm_row, t2_row in zip(branches, y_llm, y_t2):
            diff_row = llm_row - t2_row
            w.writerow([br] + llm_row.tolist() + t2_row.tolist() + diff_row.tolist())
    print(f"CSV saved → {path}\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze ChatGPT labels vs TacticalPersonality2 outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data/labeled'),
                        help='Directory containing */labels.jsonl (default: data/labeled)')
    parser.add_argument('--csv', type=Path, default=None,
                        help='Optional path to export full row-level CSV')
    args = parser.parse_args()

    print(f"\nLoading labels from {args.data_dir}/")
    X, y_llm, names = load_data(args.data_dir)
    n_scenarios = len(set(names))
    print(f"{len(X)} states across {n_scenarios} scenario(s)\n")

    y_t2 = tactical2_pred(X)

    # ── Overall stats ─────────────────────────────────────────────────────────
    print(SEP2)
    print("  OVERVIEW — mean weights per module")
    print(SEP2)
    print_overview(X, y_llm, y_t2)

    # ── Per-branch ────────────────────────────────────────────────────────────
    print(SEP2)
    print("  BRANCH BREAKDOWN — per tactical situation")
    print(SEP2)
    print()
    print_branch_breakdown(X, y_llm, y_t2)

    # ── Bias summary ──────────────────────────────────────────────────────────
    print(SEP2)
    print("  SYSTEMATIC BIAS — where LLM consistently leans vs T2  (threshold ±0.05)")
    print(SEP2)
    print()
    print_bias_summary(y_llm, y_t2)

    # ── Correlation ───────────────────────────────────────────────────────────
    print(SEP2)
    print("  CORRELATION — how well LLM tracks T2 per module")
    print(SEP2)
    print()
    print_correlation(y_llm, y_t2)

    # ── Worst disagreements ───────────────────────────────────────────────────
    print(SEP2)
    print("  TOP 10 WORST DISAGREEMENTS (by summed |Δ| across tactical modules)")
    print(SEP2)
    print()
    print_worst_disagreements(X, y_llm, y_t2, top_n=10)

    if args.csv:
        export_csv(X, y_llm, y_t2, args.csv)


if __name__ == '__main__':
    main()
