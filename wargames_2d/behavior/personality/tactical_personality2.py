from behavior.personality import Personality, ShipState


class TacticalPersonality2(Personality):
    """
    Tactical 2.0 — full rewrite based on the ChatGPT labeling prompt guidelines
    (label_with_chatgpt.py build_system_prompt).

    Decision tree:
      A. HP < 30%       → survival override: max evasion, abandon everything else
      B. Reloading
           B1. Fully shielded (≥85%) + healthy (≥80%) → hold formation, slight attack
           B2. Otherwise → retreat, evasion dominant, AttackModule = 0
               (long-reload ships retreat more aggressively)
      C. Loaded, can_fire → lock mode: AttackModule dominant, suppress other modules
      D. Loaded, has_target, no lock → closing mode: charge in, break formation
      E. Loaded, no target → ObjectiveModule dominant, hold formation

    Post-branch modifiers (kill pressure, force ratio, objective hold,
    shield depletion, hit-and-run).

    BoundsModule and CollisionAvoidanceModule are intentionally NOT overridden —
    their static defaults satisfy the ≥ 0.5 hard minimum from the prompt.
    """

    def get_weights(self, state: ShipState) -> dict[str, float]:
        fast_ship   = state.max_speed_stat >= 115.0
        long_reload = state.reload_duration >= 7.0

        # ── A: HP critical — survival overrides mission ────────────────────────
        if state.hp_frac < 0.30:
            return _clamp({
                'EvasionModule':   1.0,
                'ObjectiveModule': 0.0,
                'AttackModule':    0.0,
                'FlockingModule':  0.1 if state.friend_count > 1 else 0.0,
            })

        # ── B: Reloading ───────────────────────────────────────────────────────
        if not state.weapon_loaded:
            urgency = 1.0 - state.reload_progress   # 1 = just started, 0 = nearly done

            if state.shield_frac >= 0.85 and state.hp_frac >= 0.80:
                # B1: Absorb hits, hold attack formation — shield will cover the reload
                objective = 0.6
                flock     = 0.8
                evasion   = 0.3
                attack    = 0.3
            else:
                # B2: No shield / damaged — retreat behind allies
                retreat   = 0.3 if long_reload else 0.0
                evasion   = min(0.7 + 0.2 * urgency + retreat, 1.0)
                attack    = 0.0
                objective = 0.4
                flock     = 0.3 if state.friend_count > 1 else 0.1

            return _clamp({
                'ObjectiveModule': objective,
                'FlockingModule':  flock,
                'EvasionModule':   evasion,
                'AttackModule':    attack,
            })

        # ── Loaded: shared force-ratio flags ──────────────────────────────────
        outnumbered = state.enemy_count > state.friend_count + 1
        isolated    = state.friend_count <= 1
        dominant    = (state.friend_count + 1) >= 2 * max(state.enemy_count, 1)

        if state.has_target:
            if state.can_fire:
                # ── C: Lock mode — in firing position, hold the lock and fire ──
                objective = 0.4 if state.inside_objective else 0.1
                flock     = 0.2
                evasion   = 0.1
                attack    = 1.0

                if outnumbered:
                    evasion = 0.3 if isolated else 0.2
                    flock   = 0.1 if isolated else 0.4

            else:
                # ── D: Closing mode — charge to establish firing position ──────
                objective = 0.4
                flock     = 0.3   # some cohesion but don't stall the attack run
                evasion   = 0.1
                attack    = 1.0

                # Outnumbered + isolated: take cover instead of solo charge
                if outnumbered and isolated:
                    evasion = 0.4
                    attack  = 0.6
                    flock   = 0.1

                # Hit-and-run: fast + unshielded + ammo almost spent → break off
                if fast_ship and state.shield_frac == 0.0 and state.ammo_frac < 0.30:
                    attack  = 0.5
                    evasion = 0.7
                    flock   = 0.1

        else:
            # ── E: No target — navigate to objective and hold formation ─────────
            objective = 0.8
            flock     = 0.5
            evasion   = 0.1
            attack    = 0.3   # converge toward enemies even without a lock

        # ── Post-branch modifiers ─────────────────────────────────────────────

        # Kill pressure: finish a nearly-dead target
        if state.has_target and state.target_hp_frac < 0.20:
            attack  = min(attack  + 0.2, 1.0)
            evasion = max(evasion - 0.1, 0.0)

        # Objective hold reinforcement
        if state.inside_objective:
            objective = min(objective + 0.2, 1.0)

        # Dominant force: push harder across the board
        if dominant and state.has_target:
            attack  = min(attack  + 0.1, 1.0)
            evasion = max(evasion - 0.1, 0.0)

        # Outnumbered with allies nearby: form up
        if outnumbered and not isolated:
            flock = min(flock + 0.2, 1.0)

        # Shield nearly depleted and HP not full: take cover
        if state.shield_frac < 0.15 and state.hp_frac < 0.75:
            evasion = min(evasion + 0.2, 1.0)

        return _clamp({
            'ObjectiveModule': objective,
            'FlockingModule':  flock,
            'EvasionModule':   evasion,
            'AttackModule':    attack,
        })


def _clamp(weights: dict) -> dict:
    return {k: max(0.0, min(1.0, v)) for k, v in weights.items()}
