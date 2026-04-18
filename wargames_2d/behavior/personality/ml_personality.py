"""
MLPersonality — runs a trained micro-model to produce steering weights.

Loads a two-layer MLP stored in a .npz file and uses it to predict per-module
weights from a ShipState snapshot.  Activated in scenarios via:

    "personality": "ml:path/to/model.npz"

Expected .npz keys:
    W1  float32 (H, 20)   — first layer weights
    b1  float32 (H,)      — first layer biases
    W2  float32 (4, H)    — second layer weights
    b2  float32 (4,)      — second layer biases

Architecture: Linear(20, H) → Tanh → Linear(H, 4) → Sigmoid
Outputs 4 tactical modules only. BoundsModule and CollisionAvoidanceModule
are static weights set by the spawner and are not part of the model output.
Recommended H = 32  →  712 total parameters.
"""

import numpy as np

from behavior.personality import Personality, ShipState, state_to_vector


# Tactical modules only — order must match train_model.py MODULE_ORDER
_ML_MODULE_NAMES = [
    'ObjectiveModule',
    'FlockingModule',
    'EvasionModule',
    'AttackModule',
]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class MLPersonality(Personality):
    """
    Two-layer MLP personality loaded from a .npz weights file.

    The model maps a 20-float state vector to 4 tactical module weights in [0, 1].
    BoundsModule and CollisionAvoidanceModule are not returned — the game engine
    applies their static defaults.  Inference is pure numpy — no framework dependencies.
    """

    def __init__(self, weights_path: str) -> None:
        data    = np.load(weights_path)
        self.W1 = data['W1']   # (H, 20)
        self.b1 = data['b1']   # (H,)
        self.W2 = data['W2']   # (4, H)
        self.b2 = data['b2']   # (4,)

    def get_weights(self, state: ShipState) -> dict[str, float]:
        x   = state_to_vector(state)          # (20,)
        h   = np.tanh(self.W1 @ x + self.b1)  # (H,)
        out = _sigmoid(self.W2 @ h + self.b2)  # (4,)
        return dict(zip(_ML_MODULE_NAMES, out.tolist()))
