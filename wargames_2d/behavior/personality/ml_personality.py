"""
MLPersonality — runs a trained micro-model to produce steering weights.

Loads a two-layer MLP stored in a .npz file and uses it to predict per-module
weights from a ShipState snapshot.  Activated in scenarios via:

    "personality": "ml:path/to/model.npz"

Expected .npz keys:
    W1  float32 (H, 20)   — first layer weights
    b1  float32 (H,)      — first layer biases
    W2  float32 (6, H)    — second layer weights
    b2  float32 (6,)      — second layer biases

Architecture: Linear(20, H) → Tanh → Linear(H, 6) → Sigmoid
Recommended H = 32  →  742 total parameters.
"""

import numpy as np

from behavior.personality import Personality, ShipState, state_to_vector, MODULE_NAMES


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class MLPersonality(Personality):
    """
    Two-layer MLP personality loaded from a .npz weights file.

    The model maps a 20-float state vector to 6 module weights in [0, 1].
    Inference is pure numpy — no framework dependencies.
    """

    def __init__(self, weights_path: str) -> None:
        data    = np.load(weights_path)
        self.W1 = data['W1']   # (H, 20)
        self.b1 = data['b1']   # (H,)
        self.W2 = data['W2']   # (6, H)
        self.b2 = data['b2']   # (6,)

    def get_weights(self, state: ShipState) -> dict[str, float]:
        x   = state_to_vector(state)          # (20,)
        h   = np.tanh(self.W1 @ x + self.b1)  # (H,)
        out = _sigmoid(self.W2 @ h + self.b2)  # (6,)
        return dict(zip(MODULE_NAMES, out.tolist()))
