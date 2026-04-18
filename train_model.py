#!/usr/bin/env python3
"""
train_model.py — Train a micro-model MLP on ChatGPT-labeled ship state data.

Loads data/labeled/*/labels.jsonl, trains a two-layer MLP using MLX (Apple Silicon GPU),
and exports weights to a .npz file for pure-numpy runtime inference in the game.

Architecture: Linear(20, H) → Tanh → Linear(H, 6) → Sigmoid
Default H = 32  →  870 total parameters.

Usage:
    pip install mlx numpy
    python train_model.py
    python train_model.py --hidden 48 --epochs 1000 --output models/ship_brain_v2.npz
"""

import argparse
import json
import random
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

# ── Tactical modules only — BoundsModule and CollisionAvoidanceModule are
#    static weights in the game engine and excluded from the model. ────────────
MODULE_ORDER = [
    'ObjectiveModule',
    'FlockingModule',
    'EvasionModule',
    'AttackModule',
]
N_INPUTS  = 20
N_OUTPUTS = len(MODULE_ORDER)  # 4


# ── Model ────────────────────────────────────────────────────────────────────────

class ShipBrainMLP(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(N_INPUTS, hidden)
        self.fc2 = nn.Linear(hidden, N_OUTPUTS)

    def __call__(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.fc2(mx.tanh(self.fc1(x))))


# ── Data loading ─────────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load all labels.jsonl under data_dir.  Returns X (N,15), y (N,6) float32."""
    X_rows, y_rows = [], []
    files = sorted(data_dir.glob('*/labels.jsonl'))
    if not files:
        raise FileNotFoundError(f"No labels.jsonl files found under {data_dir}")

    for path in files:
        n = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                X_rows.append(record['state'])
                y_rows.append([record['weights'][m] for m in MODULE_ORDER])
                n += 1
        print(f"  {path.parent.name}: {n} rows")

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows,  dtype=np.float32)
    return X, y


# ── TacticalPersonality2 baseline ────────────────────────────────────────────────

def tactical2_pred(X: np.ndarray) -> np.ndarray:
    """
    Reconstruct TacticalPersonality2 outputs from state vectors.

    Feature indices used (see behavior/personality/__init__.py state_to_vector):
      0  hp_frac          4  reload_progress   10 can_fire
      1  shield_frac      5  speed_frac        11 friend_count (n/30)
      2  weapon_loaded    6  has_target        12 enemy_count  (n/30)
      3  ammo_frac        7  target_distance   14 inside_objective
                          8  target_hp_frac    16 max_speed_stat/130
                          9  closing_speed     19 reload_duration/11
    """
    N   = len(X)
    idx = {m: i for i, m in enumerate(MODULE_ORDER)}
    y   = np.zeros((N, N_OUTPUTS), dtype=np.float32)

    def _clamp(v):
        return max(0.0, min(1.0, v))

    for i, r in enumerate(X):
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
            y[i, idx['EvasionModule']]  = 1.0
            y[i, idx['FlockingModule']] = 0.1 if friend_count > 1 else 0.0
            continue

        # B: Reloading
        if not weapon_loaded:
            urgency = 1.0 - reload_progress
            if shield_frac >= 0.85 and hp_frac >= 0.80:
                objective, flock, evasion, attack = 0.6, 0.8, 0.3, 0.3
            else:
                retreat = 0.3 if long_reload else 0.0
                evasion = min(0.7 + 0.2 * urgency + retreat, 1.0)
                attack  = 0.0
                objective = 0.4
                flock = 0.3 if friend_count > 1 else 0.1
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

        y[i, idx['ObjectiveModule']] = _clamp(objective)
        y[i, idx['FlockingModule']]  = _clamp(flock)
        y[i, idx['EvasionModule']]   = _clamp(evasion)
        y[i, idx['AttackModule']]    = _clamp(attack)

    return y


# ── DefaultPersonality baseline ──────────────────────────────────────────────────

def default_personality_pred(X: np.ndarray) -> np.ndarray:
    """
    Reconstruct DefaultPersonality outputs from state vectors (tactical modules only).
    Feature index 2 = weapon_loaded (1.0 = loaded, 0.0 = reloading).
    """
    N = len(X)
    y_pred = np.zeros((N, N_OUTPUTS), dtype=np.float32)
    idx = {m: i for i, m in enumerate(MODULE_ORDER)}

    for i, row in enumerate(X):
        weapon_loaded = row[2] > 0.5
        if not weapon_loaded:
            y_pred[i, idx['EvasionModule']]   = 0.9
            y_pred[i, idx['AttackModule']]    = 0.0
            y_pred[i, idx['ObjectiveModule']] = 0.0
            y_pred[i, idx['FlockingModule']]  = 0.05
        else:
            y_pred[i, idx['AttackModule']]    = 1.0
            y_pred[i, idx['EvasionModule']]   = 0.3
            y_pred[i, idx['ObjectiveModule']] = 0.5
            y_pred[i, idx['FlockingModule']]  = 0.5
    return y_pred


# ── Training ──────────────────────────────────────────────────────────────────────

def loss_fn(model, X: mx.array, y: mx.array, weight_decay: float) -> mx.array:
    preds = model(X)
    mse   = mx.mean((preds - y) ** 2)
    l2    = weight_decay * (mx.sum(model.fc1.weight ** 2) + mx.sum(model.fc2.weight ** 2))
    return mse + l2


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    hidden:       int   = 32,
    epochs:       int   = 500,
    lr:           float = 1e-3,
    weight_decay: float = 1e-3,
) -> ShipBrainMLP:
    model     = ShipBrainMLP(hidden=hidden)
    optimizer = optim.Adam(learning_rate=lr)

    Xt = mx.array(X_train)
    yt = mx.array(y_train)

    loss_and_grad = nn.value_and_grad(model, lambda m, x, y_: loss_fn(m, x, y_, weight_decay))

    print(f"\nTraining  hidden={hidden}  epochs={epochs}  lr={lr}  wd={weight_decay}")
    print(f"  Train rows: {len(X_train)}  Val rows: {len(X_val)}")
    print(f"  Params: {hidden*N_INPUTS + hidden + N_OUTPUTS*hidden + N_OUTPUTS}")
    print()

    for epoch in range(1, epochs + 1):
        loss, grads = loss_and_grad(model, Xt, yt)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        if epoch % 50 == 0 or epoch == 1:
            Xv = mx.array(X_val)
            yv = mx.array(y_val)
            val_preds = model(Xv)
            val_mse   = float(mx.mean((val_preds - yv) ** 2).item())
            print(f"  epoch {epoch:4d}  train_loss={float(loss.item()):.5f}  val_mse={val_mse:.5f}")

    return model


# ── Evaluation ────────────────────────────────────────────────────────────────────

def evaluate(model: ShipBrainMLP, X_val: np.ndarray, y_val: np.ndarray) -> None:
    Xv     = mx.array(X_val)
    preds  = np.array(model(Xv))

    mse    = float(np.mean((preds - y_val) ** 2))
    mae    = float(np.mean(np.abs(preds - y_val)))

    # Baselines
    t2_pred = tactical2_pred(X_val)
    t2_mse  = float(np.mean((t2_pred - y_val) ** 2))
    t2_mae  = float(np.mean(np.abs(t2_pred - y_val)))

    dp_pred = default_personality_pred(X_val)
    dp_mse  = float(np.mean((dp_pred - y_val) ** 2))
    dp_mae  = float(np.mean(np.abs(dp_pred - y_val)))

    print("\n── Validation results ─────────────────────────────────────────────────")
    print(f"  Model      MSE={mse:.5f}  MAE={mae:.4f}")
    print(f"  Tactical2  MSE={t2_mse:.5f}  MAE={t2_mae:.4f}  (teacher — near 0 if labels from tactical2)")
    print(f"  Default    MSE={dp_mse:.5f}  MAE={dp_mae:.4f}  (DefaultPersonality)")
    if mse < t2_mse:
        print("  ✓ Model beats Tactical2 baseline")
    elif mse < dp_mse:
        print("  ~ Model beats Default but not Tactical2 — needs more data or larger hidden")
    else:
        print("  ✗ Model does not beat Default baseline — consider more data or tuning")

    # Per-module MAE
    per_module_mae = np.mean(np.abs(preds - y_val), axis=0)
    print("\n  Per-module MAE:")
    for name, err in zip(MODULE_ORDER, per_module_mae):
        bar = '█' * int(err * 40)
        print(f"    {name:<28s}  {err:.4f}  {bar}")
    print()


# ── Export ────────────────────────────────────────────────────────────────────────

def export(model: ShipBrainMLP, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(output_path),
        W1=np.array(model.fc1.weight),   # (H, 15)
        b1=np.array(model.fc1.bias),     # (H,)
        W2=np.array(model.fc2.weight),   # (6, H)
        b2=np.array(model.fc2.bias),     # (6,)
    )
    print(f"Saved model → {output_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the ship-brain MLP on ChatGPT-labeled data using MLX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data-dir', type=Path, default=Path('data/labeled'),
                        help='Directory containing */labels.jsonl (default: data/labeled)')
    parser.add_argument('--output',   type=Path, default=Path('models/ship_brain.npz'),
                        help='Output .npz path (default: models/ship_brain.npz)')
    parser.add_argument('--hidden',   type=int,   default=32,   help='Hidden layer size (default: 32)')
    parser.add_argument('--epochs',   type=int,   default=500,  help='Training epochs (default: 500)')
    parser.add_argument('--lr',       type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--wd',       type=float, default=1e-3, help='Weight decay / L2 (default: 1e-3)')
    parser.add_argument('--seed',     type=int,   default=42,   help='Random seed (default: 42)')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────────────────────
    print(f"Loading data from {args.data_dir}/")
    X, y = load_data(args.data_dir)
    print(f"\nTotal: {len(X)} rows  X={X.shape}  y={y.shape}")

    # ── Split ────────────────────────────────────────────────────────────────────
    idx   = np.random.permutation(len(X))
    split = int(0.8 * len(X))
    train_idx, val_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_val,   y_val   = X[val_idx],   y[val_idx]

    # ── Train ────────────────────────────────────────────────────────────────────
    model = train(
        X_train, y_train, X_val, y_val,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────────
    evaluate(model, X_val, y_val)

    # ── Export ───────────────────────────────────────────────────────────────────
    export(model, args.output)


if __name__ == '__main__':
    main()
