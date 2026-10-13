#!/usr/bin/env python3
"""
export_model.py — Export a trained ship-brain .npz model for use in
external applications (Unity, C#, JavaScript, etc.).

Supported output formats:
  json   Self-contained JSON with weights, architecture metadata, and a
         complete normalization specification.  Any language that can
         parse JSON and implement tanh/sigmoid can run inference.
  onnx   Standard ONNX interchange file (requires: pip install onnx).
         Consumable by Unity Sentis, ML.NET, onnxruntime-web, etc.

Usage:
    python export_model.py --model models/ship_brain_v4.npz --format json
    python export_model.py --model models/ship_brain_v4.npz --format onnx
    python export_model.py --model models/ship_brain_v4.npz --format json --format onnx
    python export_model.py --model models/ship_brain_v4.npz --format json --output-dir exports/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── Module names (must match MODULE_ORDER in train_model.py) ──────────────────

OUTPUT_MODULES = [
    'ObjectiveModule',   # 0 — drive toward / hold the team objective
    'FlockingModule',    # 1 — stay in formation with friendly ships
    'EvasionModule',     # 2 — dodge and retreat from incoming fire
    'AttackModule',      # 3 — manoeuvre to establish / hold a firing solution
]

# ── Normalization constants (must match state_to_vector in
#    wargames_2d/behavior/personality/__init__.py) ─────────────────────────────

NORM_CONSTANTS = {
    'DIST_SCALE':       400.0,   # px   — hyperbolic decay scale for distances
    'SPEED_SCALE':      100.0,   # px/s — tanh scale for closing_speed
    'COUNT_SCALE':       30.0,   # ships — cap for friend/enemy counts
    'MAX_HP_REF':       265.0,   # HP   — reference maximum HP
    'MAX_SPEED_REF':    130.0,   # px/s — reference maximum ship speed
    'WEAPON_RANGE_REF': 610.0,   # px   — reference maximum weapon range
    'WEAPON_DMG_REF':    38.0,   # dmg  — reference maximum damage per shot
    'RELOAD_DUR_REF':    11.0,   # s    — reference maximum reload duration
}

# ── Feature specification ─────────────────────────────────────────────────────
#
# Each entry describes one element of the 20-float input vector.
# Fields:
#   index       : position in the vector (0–19)
#   name        : human-readable name matching ShipState field name
#   group       : logical group (health / weapon / motion / target / team /
#                 objective / profile)
#   transform   : normalisation applied to the raw value
#   output_range: expected range after normalisation
#   edge_case   : what value is used when the field is unavailable

FEATURE_SPEC = [
    # Health
    {
        'index': 0, 'group': 'health', 'name': 'hp_frac',
        'raw_description': 'current HP / max HP',
        'transform': 'identity',
        'formula': 'hp / max_hp',
        'output_range': '[0, 1]',
        'edge_case': None,
    },
    {
        'index': 1, 'group': 'health', 'name': 'shield_frac',
        'raw_description': 'current shield / max shield',
        'transform': 'identity',
        'formula': 'shield / max_shield',
        'output_range': '[0, 1]',
        'edge_case': '0.0 when ship has no shield (max_shield == 0)',
    },
    # Weapon
    {
        'index': 2, 'group': 'weapon', 'name': 'weapon_loaded',
        'raw_description': 'cassette ready to fire',
        'transform': 'bool_to_float',
        'formula': 'float(not reloading)',
        'output_range': '{0.0, 1.0}',
        'edge_case': '1.0 when ship has no weapon',
    },
    {
        'index': 3, 'group': 'weapon', 'name': 'ammo_frac',
        'raw_description': 'shots remaining in cassette / cassette_size',
        'transform': 'identity',
        'formula': '(cassette_size - shots_fired) / cassette_size',
        'output_range': '[0, 1]',
        'edge_case': '1.0 when ship has no weapon',
    },
    {
        'index': 4, 'group': 'weapon', 'name': 'reload_progress',
        'raw_description': '0 = reload just started, 1 = fully loaded',
        'transform': 'identity',
        'formula': '1.0 - reload_timer / reload_duration  (while reloading);  1.0 otherwise',
        'output_range': '[0, 1]',
        'edge_case': '1.0 when ship has no weapon',
    },
    # Motion
    {
        'index': 5, 'group': 'motion', 'name': 'speed_frac',
        'raw_description': 'current speed / max_speed',
        'transform': 'identity',
        'formula': 'velocity.magnitude / max_speed',
        'output_range': '[0, 1]',
        'edge_case': None,
    },
    # Target
    {
        'index': 6, 'group': 'target', 'name': 'has_target',
        'raw_description': 'whether an alive target is tracked',
        'transform': 'bool_to_float',
        'formula': 'float(target is not None and target.is_alive())',
        'output_range': '{0.0, 1.0}',
        'edge_case': '0.0 when no target',
    },
    {
        'index': 7, 'group': 'target', 'name': 'target_distance',
        'raw_description': 'distance to tracked target (px)',
        'transform': 'hyperbolic_decay',
        'formula': '1.0 / (1.0 + dist / 400.0)',
        'output_range': '(0, 1]',
        'edge_case': '0.0 when no target (maps infinity → 0)',
        'constant': 'DIST_SCALE',
    },
    {
        'index': 8, 'group': 'target', 'name': 'target_hp_frac',
        'raw_description': "target's current HP / target's max HP",
        'transform': 'identity',
        'formula': 'target.hp / target.max_hp',
        'output_range': '[0, 1]',
        'edge_case': '1.0 when no target',
    },
    {
        'index': 9, 'group': 'target', 'name': 'closing_speed',
        'raw_description': 'relative approach speed along the line to target (px/s, positive = closing)',
        'transform': 'tanh_normalized',
        'formula': 'tanh((ship_vel - target_vel) · to_target_unit / 100.0)',
        'output_range': '(-1, 1)',
        'edge_case': '0.0 when no target',
        'constant': 'SPEED_SCALE',
        'notes': 'dot product of (ship_velocity - target_velocity) onto the unit vector '
                 'pointing from ship toward target; positive = approaching, negative = retreating',
    },
    {
        'index': 10, 'group': 'target', 'name': 'can_fire',
        'raw_description': 'weapon is loaded, target in range, within fire arc, and LoS is clear',
        'transform': 'bool_to_float',
        'formula': 'float(loaded AND in_range AND in_arc AND clear_los)',
        'output_range': '{0.0, 1.0}',
        'edge_case': '0.0 when no target or weapon is reloading',
    },
    # Team
    {
        'index': 11, 'group': 'team', 'name': 'friend_count',
        'raw_description': 'number of alive friendly ships within visual_range',
        'transform': 'capped_ratio',
        'formula': 'min(friend_count / 30.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': None,
        'constant': 'COUNT_SCALE',
    },
    {
        'index': 12, 'group': 'team', 'name': 'enemy_count',
        'raw_description': 'number of alive enemy ships within visual_range',
        'transform': 'capped_ratio',
        'formula': 'min(enemy_count / 30.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': None,
        'constant': 'COUNT_SCALE',
    },
    # Objective
    {
        'index': 13, 'group': 'objective', 'name': 'objective_distance',
        'raw_description': 'distance to team objective centre (px)',
        'transform': 'hyperbolic_decay',
        'formula': '1.0 / (1.0 + dist / 400.0)',
        'output_range': '(0, 1]',
        'edge_case': '0.0 when no objective (maps infinity → 0)',
        'constant': 'DIST_SCALE',
    },
    {
        'index': 14, 'group': 'objective', 'name': 'inside_objective',
        'raw_description': 'ship is within the patrol ring of the team objective',
        'transform': 'bool_to_float',
        'formula': 'float(objective_distance <= patrol_radius)',
        'output_range': '{0.0, 1.0}',
        'edge_case': '0.0 when no objective',
    },
    # Ship profile (absolute stats, normalised against global reference maxima)
    {
        'index': 15, 'group': 'profile', 'name': 'max_hp',
        'raw_description': "ship's configured maximum HP",
        'transform': 'capped_ratio',
        'formula': 'min(max_hp / 265.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': None,
        'constant': 'MAX_HP_REF',
    },
    {
        'index': 16, 'group': 'profile', 'name': 'max_speed_stat',
        'raw_description': "ship's configured maximum speed (px/s)",
        'transform': 'capped_ratio',
        'formula': 'min(max_speed / 130.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': None,
        'constant': 'MAX_SPEED_REF',
    },
    {
        'index': 17, 'group': 'profile', 'name': 'weapon_range',
        'raw_description': 'weapon range (px)',
        'transform': 'capped_ratio',
        'formula': 'min(weapon_range / 610.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': '0.0 when ship has no weapon',
        'constant': 'WEAPON_RANGE_REF',
    },
    {
        'index': 18, 'group': 'profile', 'name': 'weapon_damage',
        'raw_description': 'weapon damage per shot',
        'transform': 'capped_ratio',
        'formula': 'min(damage / 38.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': '0.0 when ship has no weapon',
        'constant': 'WEAPON_DMG_REF',
    },
    {
        'index': 19, 'group': 'profile', 'name': 'reload_duration',
        'raw_description': 'weapon reload time (seconds)',
        'transform': 'capped_ratio',
        'formula': 'min(reload_duration / 11.0, 1.0)',
        'output_range': '[0, 1]',
        'edge_case': '0.0 when ship has no weapon',
        'constant': 'RELOAD_DUR_REF',
    },
]


# ── Loader ────────────────────────────────────────────────────────────────────

def load_npz(path: Path) -> dict:
    """Load .npz and return a plain dict of numpy arrays."""
    data = np.load(str(path))
    weights = {k: data[k] for k in data.files}
    required = {'W1', 'b1', 'W2', 'b2'}
    missing = required - weights.keys()
    if missing:
        raise ValueError(f"Missing keys in {path}: {missing}")
    return weights


# ── JSON export ───────────────────────────────────────────────────────────────

def export_json(weights: dict, output_path: Path) -> None:
    """
    Write a self-contained JSON file with model weights, architecture
    metadata, and normalization specification.

    The JSON can be loaded by any language that supports JSON parsing.
    All weight matrices are stored as nested lists (float32 → Python float
    for portability).
    """
    W1 = weights['W1']   # (hidden, 20)
    b1 = weights['b1']   # (hidden,)
    W2 = weights['W2']   # (4, hidden)
    b2 = weights['b2']   # (4,)

    hidden = W1.shape[0]

    payload = {
        'schema_version': '1.0',
        'description': (
            'Ship-brain micro-model: predicts per-module steering weights '
            'from a 20-float ship state vector.'
        ),

        'architecture': {
            'input_dim': int(W1.shape[1]),
            'hidden_dim': int(hidden),
            'output_dim': int(W2.shape[0]),
            'activation_hidden': 'tanh',
            'activation_output': 'sigmoid',
            'forward_pass': (
                'h = tanh(W1 @ x + b1);  out = sigmoid(W2 @ h + b2)'
            ),
            'weight_shapes': {
                'W1': list(W1.shape),
                'b1': list(b1.shape),
                'W2': list(W2.shape),
                'b2': list(b2.shape),
            },
        },

        'outputs': [
            {'index': i, 'name': name, 'description': _output_description(name)}
            for i, name in enumerate(OUTPUT_MODULES)
        ],

        'normalization': {
            'description': (
                'Apply these transforms to raw game values before passing to the model. '
                'All outputs should be in [0,1] except closing_speed which is in (-1,1).'
            ),
            'constants': NORM_CONSTANTS,
            'features': FEATURE_SPEC,
        },

        'weights': {
            'W1': W1.tolist(),
            'b1': b1.tolist(),
            'W2': W2.tolist(),
            'b2': b2.tolist(),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    print(f"JSON export → {output_path}  ({output_path.stat().st_size // 1024} KB)")


def _output_description(name: str) -> str:
    descriptions = {
        'ObjectiveModule': 'Weight for driving toward / holding the team objective area',
        'FlockingModule':  'Weight for staying in formation with friendly ships',
        'EvasionModule':   'Weight for dodging and retreating from incoming fire',
        'AttackModule':    'Weight for manoeuvring to establish / hold a firing solution',
    }
    return descriptions.get(name, '')


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(weights: dict, output_path: Path) -> None:
    """
    Write an ONNX model file.

    Graph:  Input(batch,20) → Gemm(W1,b1,transB=1) → Tanh → Gemm(W2,b2,transB=1) → Sigmoid → Output(batch,4)

    Gemm with transB=1 computes  Y = X × B^T + C, which matches the numpy
    forward pass  h = W1 @ x + b1  when B = W1 and C = b1.

    Requires: pip install onnx
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError:
        print("ERROR: 'onnx' package not installed.  Run:  pip install onnx", file=sys.stderr)
        sys.exit(1)

    W1 = weights['W1'].astype(np.float32)   # (hidden, 20)
    b1 = weights['b1'].astype(np.float32)   # (hidden,)
    W2 = weights['W2'].astype(np.float32)   # (4, hidden)
    b2 = weights['b2'].astype(np.float32)   # (4,)

    hidden   = W1.shape[0]
    in_dim   = W1.shape[1]
    out_dim  = W2.shape[0]

    # Initializers (stored weights)
    init_W1 = numpy_helper.from_array(W1, name='W1')
    init_b1 = numpy_helper.from_array(b1, name='b1')
    init_W2 = numpy_helper.from_array(W2, name='W2')
    init_b2 = numpy_helper.from_array(b2, name='b2')

    # Nodes
    #  Gemm: Y = alpha * A * B^T + beta * C   (transB=1 → B^T = W^T, so Y = X * W^T + b)
    gemm1 = helper.make_node(
        'Gemm', inputs=['input', 'W1', 'b1'], outputs=['pre_tanh'],
        transB=1, alpha=1.0, beta=1.0,
    )
    tanh_node = helper.make_node('Tanh', inputs=['pre_tanh'], outputs=['hidden'])
    gemm2 = helper.make_node(
        'Gemm', inputs=['hidden', 'W2', 'b2'], outputs=['pre_sigmoid'],
        transB=1, alpha=1.0, beta=1.0,
    )
    sigmoid_node = helper.make_node('Sigmoid', inputs=['pre_sigmoid'], outputs=['output'])

    # Graph
    graph = helper.make_graph(
        nodes=[gemm1, tanh_node, gemm2, sigmoid_node],
        name='ship_brain',
        inputs=[
            helper.make_tensor_value_info('input',  TensorProto.FLOAT, [None, in_dim]),
        ],
        outputs=[
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, out_dim]),
        ],
        initializer=[init_W1, init_b1, init_W2, init_b2],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)])
    model.doc_string = (
        'Ship-brain micro-model.  Input: (batch, 20) float32 normalised state vector.  '
        'Output: (batch, 4) float32 steering weights [Objective, Flocking, Evasion, Attack].'
    )
    onnx.checker.check_model(model)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output_path))
    print(f"ONNX export → {output_path}  ({output_path.stat().st_size} bytes)")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Export a trained ship-brain .npz model to JSON / ONNX.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--model', type=Path, default=Path('models/ship_brain_v4.npz'),
        help='Path to the source .npz file (default: models/ship_brain_v4.npz)',
    )
    parser.add_argument(
        '--format', dest='formats', action='append', choices=['json', 'onnx'],
        metavar='FORMAT',
        help='Export format: json and/or onnx  (repeat flag for multiple formats)',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('.'),
        help='Directory to write exported files (default: current directory)',
    )
    args = parser.parse_args()

    formats = args.formats or ['json']   # default to JSON if no --format given

    # Load weights
    if not args.model.exists():
        print(f"ERROR: model file not found: {args.model}", file=sys.stderr)
        sys.exit(1)
    weights = load_npz(args.model)
    W1 = weights['W1']
    print(f"Loaded {args.model}  — architecture: Linear({W1.shape[1]}, {W1.shape[0]}) → Tanh "
          f"→ Linear({W1.shape[0]}, {weights['W2'].shape[0]}) → Sigmoid")

    stem = args.model.stem   # e.g. 'ship_brain_v4'

    if 'json' in formats:
        export_json(weights, args.output_dir / f'{stem}.json')

    if 'onnx' in formats:
        export_onnx(weights, args.output_dir / f'{stem}.onnx')


if __name__ == '__main__':
    main()
