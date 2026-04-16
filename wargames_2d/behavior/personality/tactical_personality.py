from behavior.personality import Personality, ShipState


class TacticalPersonality(Personality):
    """
    Multi-factor rule-based personality designed as a strong baseline for
    comparison against DefaultPersonality and MLPersonality.

    Six sequential modifier passes:
      1. HP pressure        — retreat progressively as health drops
      2. Shield depletion   — extra caution when shields are gone and HP is low
      3. Weapon reload      — suppress attack, taper evasion toward end of reload,
                              maintain minimum objective pressure (unlike DefaultPersonality)
      4. Force ratio        — form up when outnumbered, push when dominant
      5. Engagement         — commit to attack when in range, finish low-HP targets
      6. Objective hold     — reinforce objective weight when already inside

    BoundsModule and CollisionAvoidanceModule are left at their static defaults
    (safety-critical, never overridden).
    """

    def get_weights(self, state: ShipState) -> dict[str, float]:
        # --- Base weights (balanced, healthy, loaded, even forces) ---
        objective = 0.7
        flock     = 0.2
        evasion   = 0.35
        attack    = 0.8

        # --- Pass 1: HP pressure ---
        if state.hp_frac < 0.25:
            # Critical: survival mode
            evasion   = evasion + 0.5
            attack    = attack * 0.2
            objective = objective * 0.5
            flock     = flock + 0.3
        elif state.hp_frac < 0.50:
            # Damaged: defensive mode
            evasion   = evasion + 0.25
            attack    = attack * 0.6

        # --- Pass 2: Shield depleted ---
        if state.shield_frac < 0.15 and state.hp_frac < 0.75:
            evasion = evasion + 0.1

        # --- Pass 3: Weapon reload ---
        if not state.weapon_loaded:
            # Evasion urgency tapers as reload_progress approaches 1
            evasion   = evasion + 0.3 * (1.0 - state.reload_progress)
            attack    = 0.0
            # Keep minimum objective pressure — don't fully abandon the mission
            objective = max(objective * 0.5, 0.25)
            flock     = flock + 0.15

        # --- Pass 4: Force ratio ---
        ratio = (state.friend_count + 1) / max(state.enemy_count, 1)
        if ratio < 0.5:
            # Heavily outnumbered: tighten formation, be defensive
            flock   = flock + 0.3
            evasion = evasion + 0.15
            attack  = attack * 0.75
        elif ratio >= 2.0:
            # Dominant force: push aggressively
            attack    = attack * 1.25
            evasion   = evasion - 0.1
            objective = objective * 1.15

        # --- Pass 5: Engagement situation ---
        if state.has_target and state.weapon_loaded:
            if state.can_fire:
                # In range and ready: commit to the shot
                attack    = attack * 1.1
                evasion   = evasion - 0.1
                objective = max(objective * 0.7, 0.3)

            if state.target_hp_frac < 0.3:
                # Target nearly dead: finish the kill
                attack  = attack + 0.15
                evasion = evasion - 0.1

        # --- Pass 6: Objective hold ---
        if state.inside_objective:
            objective = objective + 0.1

        # --- Final clamp to [0, 1] ---
        return {
            'ObjectiveModule': max(0.0, min(objective, 1.0)),
            'FlockingModule':  max(0.0, min(flock,     1.0)),
            'EvasionModule':   max(0.0, min(evasion,   1.0)),
            'AttackModule':    max(0.0, min(attack,    1.0)),
        }
