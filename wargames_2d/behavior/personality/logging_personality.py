"""
LoggingPersonality — records every (state_vector, weights) call for data collection.

Wraps any inner Personality so the simulation runs normally while all per-ship
per-tick state snapshots and the corresponding weight decisions are accumulated
in memory.  After the run, call to_arrays() to extract numpy arrays ready for
saving to disk or feeding to a training script.
"""

import numpy as np

from behavior.personality import Personality, ShipState, state_to_vector, MODULE_NAMES


class LoggingPersonality(Personality):
    """
    Transparent wrapper around an inner Personality that logs every call.

    Usage in a headless data-collection run:

        logger = LoggingPersonality(DefaultPersonality())
        for ship in ShipRegistry.ships:
            ship.brain.personality = logger   # hot-swap all ships

        run_scenario(config)                  # simulation populates logger.records

        states, weights = logger.to_arrays()  # shape (N, 20) and (N, 6)
    """

    def __init__(self, inner: Personality) -> None:
        self.inner = inner
        self.records: list[tuple[np.ndarray, list[float]]] = []

    # ------------------------------------------------------------------

    def get_weights(self, state: ShipState) -> dict[str, float]:
        weights = self.inner.get_weights(state)
        vec = state_to_vector(state)
        # Build an ordered weight vector aligned with MODULE_NAMES.
        # Modules not returned by the inner personality keep 0.0 here;
        # the brain applies its own static defaults at runtime, but for
        # training we record only what the personality explicitly decided.
        w_vec = [weights.get(m, 0.0) for m in MODULE_NAMES]
        self.records.append((vec, w_vec))
        return weights

    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Discard all accumulated records (e.g. between episodes)."""
        self.records.clear()

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (states, weights) as float32 arrays.

        states  — shape (N, 20): state_to_vector output per call
        weights — shape (N,  6): inner personality weights in MODULE_NAMES order
        """
        if not self.records:
            return (
                np.zeros((0, 20), dtype=np.float32),
                np.zeros((0,  6), dtype=np.float32),
            )
        states  = np.array([r[0] for r in self.records], dtype=np.float32)
        weights = np.array([r[1] for r in self.records], dtype=np.float32)
        return states, weights
