import argparse
import glob
import math
import os
import sys
import pygame
from registry import ShipRegistry
from scenario import load_scenario
from simulation import setup_scenario, run_scenario
from factions import get_color
from entities.obstacle import Obstacle
from behavior.ship_brain import _MODULE_KEY

WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
FPS = 60
_CX = WINDOW_WIDTH  / 2   # screen centre x — zoom pivot
_CY = WINDOW_HEIGHT / 2   # screen centre y — zoom pivot

# Zoom
ZOOM_DEFAULT = 1.0
ZOOM_MIN     = 0.15
ZOOM_MAX     = 4.0
ZOOM_SPEED   = 1.8    # multiplicative factor per second when key held

# Demo mode
_BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DEMO_TRANSITION_MS = 4000   # ms to show result screen before loading next scenario

# Team progress bars (bottom of screen)
_TEAM_PANEL_H = 108   # total height of the team bar strip
_TEAM_PAD     = 8     # inner padding
_TEAM_LABEL_W = 56    # width reserved for row labels ("Attack", "Objective", …)
_TEAM_VAL_W   = 36    # width reserved for numeric value after the bar
_TEAM_BAR_H   = 11    # bar height
_TEAM_BAR_GAP = 3     # gap between bars

_PERSONALITY_LABELS = {
    'MLPersonality':       'AI',
    'DefaultPersonality':  'Hardcoded',
    'TacticalPersonality': 'Tactical',
}

# Weight overlay appearance
_OVERLAY_X = 10
_OVERLAY_Y = 35
_OVERLAY_W = 200
_BAR_H = 14
_BAR_GAP = 4
_LABEL_W = 120
_BAR_FILL_W = 72       # max fill bar width in pixels
_OVERLAY_PAD = 8
_SMOOTH_RATE = 8.0     # exponential smoothing rate (higher = faster to respond)

# Short display names for modules (class name → label)
_MODULE_LABELS = {
    'ObjectiveModule':          'Objective',
    'BoundsModule':             'Bounds',
    'CollisionAvoidanceModule': 'CollAvoid',
    'SeparationModule':         'Separation',
    'AlignmentModule':          'Alignment',
    'CohesionModule':           'Cohesion',
    'EvasionModule':            'Evasion',
    'AttackModule':             'Attack',
    'FiringSolutionModule':     'Attack',
}


def world_to_screen(pos: pygame.math.Vector2, cam: pygame.math.Vector2, zoom: float) -> tuple:
    sx = (pos.x + cam.x - _CX) * zoom + _CX
    sy = (pos.y + cam.y - _CY) * zoom + _CY
    return (int(sx), int(sy))


def screen_to_world(sx: int, sy: int, cam: pygame.math.Vector2, zoom: float) -> pygame.math.Vector2:
    return pygame.math.Vector2(
        (sx - _CX) / zoom + _CX - cam.x,
        (sy - _CY) / zoom + _CY - cam.y,
    )


def _draw_weight_overlay(surface, font, ship, smoothed_weights: dict, dt: float):
    """Draw the per-module weight overlay for the selected ship."""
    if ship.brain is None:
        return

    overrides = ship.brain.overrides
    modules = ship.brain.modules

    # Update smoothed weights
    for mod in modules:
        name = type(mod).__name__
        key = _MODULE_KEY.get(name, name)
        target = overrides.get(key, mod.weight)
        if not math.isfinite(target):
            target = 0.0
        target = max(0.0, target)
        prev = smoothed_weights.get(name, target)
        # Exponential smoothing: fast when far, stable when close
        alpha = 1.0 - math.exp(-_SMOOTH_RATE * dt)
        smoothed_weights[name] = prev + (target - prev) * alpha

    # Layout — extra rows at the bottom for HP and shield stats
    n = len(modules)
    _STATS_ROWS = 2 if ship.max_shield > 0 else 1   # HP always; shield only if present
    panel_h = _OVERLAY_PAD * 2 + n * (_BAR_H + _BAR_GAP) - _BAR_GAP + 20 + _STATS_ROWS * (_BAR_H + _BAR_GAP) + 6
    panel_rect = pygame.Rect(_OVERLAY_X - _OVERLAY_PAD, _OVERLAY_Y,
                             _OVERLAY_W + _OVERLAY_PAD, panel_h)

    # Background
    bg = pygame.Surface((panel_rect.w, panel_rect.h), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 160))
    surface.blit(bg, panel_rect.topleft)

    # Header
    header = font.render("Module Weights", True, (200, 200, 200))
    surface.blit(header, (_OVERLAY_X, _OVERLAY_Y + _OVERLAY_PAD))

    y = _OVERLAY_Y + _OVERLAY_PAD + 18

    for mod in modules:
        name = type(mod).__name__
        label = _MODULE_LABELS.get(name, name[:9])
        w = smoothed_weights.get(name, 0.0)
        w_clamped = max(0.0, min(1.0, w))

        # Module name
        lbl_surf = font.render(label, True, (180, 180, 180))
        surface.blit(lbl_surf, (_OVERLAY_X, y))

        # Bar background
        bar_x = _OVERLAY_X + _LABEL_W
        bar_rect = pygame.Rect(bar_x, y + 1, _BAR_FILL_W, _BAR_H - 2)
        pygame.draw.rect(surface, (40, 40, 40), bar_rect)

        # Bar fill — color shifts green→yellow→red by value
        fill_w = int(_BAR_FILL_W * w_clamped)
        if fill_w > 0:
            r = int(255 * (1.0 - w_clamped))
            g = int(220 * w_clamped)
            fill_color = (r, g, 40)
            pygame.draw.rect(surface, fill_color,
                             pygame.Rect(bar_x, y + 1, fill_w, _BAR_H - 2))

        # Border
        pygame.draw.rect(surface, (80, 80, 80), bar_rect, 1)

        # Numeric value (right-aligned in bar area)
        val_str = f"{w:.2f}"
        val_surf = font.render(val_str, True, (160, 160, 160))
        surface.blit(val_surf, (bar_x + _BAR_FILL_W + 3, y))

        y += _BAR_H + _BAR_GAP

    # --- Ship stats footer ---
    y += 4  # small divider gap

    # HP row
    hp_frac = max(0.0, min(1.0, ship.hp / max(ship.max_hp, 1e-6)))
    lbl_surf = font.render("HP", True, (180, 180, 180))
    surface.blit(lbl_surf, (_OVERLAY_X, y))
    bar_x = _OVERLAY_X + _LABEL_W
    bar_rect = pygame.Rect(bar_x, y + 1, _BAR_FILL_W, _BAR_H - 2)
    pygame.draw.rect(surface, (40, 40, 40), bar_rect)
    fill_w = int(_BAR_FILL_W * hp_frac)
    if fill_w > 0:
        pygame.draw.rect(surface, (0, 200, 60),
                         pygame.Rect(bar_x, y + 1, fill_w, _BAR_H - 2))
    pygame.draw.rect(surface, (80, 80, 80), bar_rect, 1)
    val_surf = font.render(f"{ship.hp:.0f}", True, (160, 160, 160))
    surface.blit(val_surf, (bar_x + _BAR_FILL_W + 3, y))
    y += _BAR_H + _BAR_GAP

    # Shield row (only when the ship has a shield pool)
    if ship.max_shield > 0:
        sh_frac = max(0.0, min(1.0, ship.shield / ship.max_shield))
        lbl_surf = font.render("Shield", True, (180, 180, 180))
        surface.blit(lbl_surf, (_OVERLAY_X, y))
        bar_rect = pygame.Rect(bar_x, y + 1, _BAR_FILL_W, _BAR_H - 2)
        pygame.draw.rect(surface, (10, 30, 80), bar_rect)
        fill_w = int(_BAR_FILL_W * sh_frac)
        if fill_w > 0:
            pygame.draw.rect(surface, (0, 160, 220),
                             pygame.Rect(bar_x, y + 1, fill_w, _BAR_H - 2))
        pygame.draw.rect(surface, (80, 80, 80), bar_rect, 1)
        val_surf = font.render(f"{ship.shield:.0f}", True, (160, 160, 160))
        surface.blit(val_surf, (bar_x + _BAR_FILL_W + 3, y))


def _compute_avg_weights(ships: list) -> dict:
    """Return the faction-averaged effective weight for the 4 personality-facing keys."""
    keys = ('FlockingModule', 'AttackModule', 'EvasionModule', 'ObjectiveModule')
    totals = {k: 0.0 for k in keys}
    counts = {k: 0   for k in keys}
    for ship in ships:
        if ship.brain is None:
            continue
        # Build per-ship default weights from the module list
        default_w: dict = {}
        for mod in ship.brain.modules:
            k = _MODULE_KEY.get(type(mod).__name__, type(mod).__name__)
            if k in keys and k not in default_w:
                default_w[k] = mod.weight
        for k in keys:
            eff = ship.brain.overrides.get(k, default_w.get(k, 0.0))
            if math.isfinite(eff) and eff >= 0.0:
                totals[k] += eff
                counts[k] += 1
    return {k: (totals[k] / counts[k] if counts[k] > 0 else 0.0) for k in keys}


def _draw_team_bars(surface, font, faction_initial_hp: dict) -> None:
    """Draw a per-faction status strip at the bottom of the screen."""
    factions = sorted(ShipRegistry.ships_by_faction.keys())
    if not factions:
        return

    n_factions = len(factions)
    panel_w = WINDOW_WIDTH // n_factions
    panel_y = WINDOW_HEIGHT - _TEAM_PANEL_H

    # Semi-transparent background for the whole strip
    bg = pygame.Surface((WINDOW_WIDTH, _TEAM_PANEL_H), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 170))
    surface.blit(bg, (0, panel_y))

    for i, fid in enumerate(factions):
        px = i * panel_w
        color = get_color(fid)
        ships = [s for s in ShipRegistry.ships_by_faction.get(fid, []) if s.is_alive()]

        # Personality label (read from the first living ship)
        pers_label = 'Static'
        for s in ships:
            if s.brain and s.brain.personality is not None:
                pname = type(s.brain.personality).__name__
                pers_label = _PERSONALITY_LABELS.get(pname, pname)
                break

        y = panel_y + _TEAM_PAD
        bar_x = px + _TEAM_PAD + _TEAM_LABEL_W
        bar_w = panel_w - _TEAM_PAD * 2 - _TEAM_LABEL_W - _TEAM_VAL_W

        # Header: "Faction N — Hardcoded"
        header = font.render(f"Faction {fid}  {pers_label}", True, color)
        surface.blit(header, (px + _TEAM_PAD, y))
        y += 18

        # HP bar
        current_hp  = sum(s.hp for s in ships)
        initial_hp  = faction_initial_hp.get(fid, max(current_hp, 1))
        hp_frac     = current_hp / max(initial_hp, 1.0)
        _draw_hbar(surface, font, bar_x, y, bar_w, hp_frac,
                   (0, 180, 60), "HP", f"{int(current_hp)}", px)
        y += _TEAM_BAR_H + _TEAM_BAR_GAP

        # Weight bars
        avg = _compute_avg_weights(ships)
        for label, key in (
            ('Flock',     'FlockingModule'),
            ('Attack',    'AttackModule'),
            ('Evade',     'EvasionModule'),
            ('Objective', 'ObjectiveModule'),
        ):
            w = avg.get(key, 0.0)
            wc = max(0.0, min(1.0, w))
            bar_color = (int(255 * (1.0 - wc)), int(220 * wc), 40)
            _draw_hbar(surface, font, bar_x, y, bar_w, wc,
                       bar_color, label, f"{w:.2f}", px)
            y += _TEAM_BAR_H + _TEAM_BAR_GAP


def _draw_hbar(surface, font, bar_x, y, bar_w, frac, fill_color,
               label: str, val_str: str, panel_x: int) -> None:
    """Draw a single labeled horizontal bar."""
    lbl_surf = font.render(label, True, (180, 180, 180))
    surface.blit(lbl_surf, (panel_x + _TEAM_PAD, y))
    bar_rect = pygame.Rect(bar_x, y + 1, bar_w, _TEAM_BAR_H - 2)
    pygame.draw.rect(surface, (40, 40, 40), bar_rect)
    fill_w = int(bar_w * max(0.0, min(1.0, frac)))
    if fill_w > 0:
        pygame.draw.rect(surface, fill_color,
                         pygame.Rect(bar_x, y + 1, fill_w, _TEAM_BAR_H - 2))
    pygame.draw.rect(surface, (80, 80, 80), bar_rect, 1)
    val_surf = font.render(val_str, True, (160, 160, 160))
    surface.blit(val_surf, (bar_x + bar_w + 3, y))


def discover_scenarios(base_dir: str) -> list[str]:
    """Return all scenario paths: top-level *.json then generated/*/scenario.json (alphabetical)."""
    paths: list[str] = []
    paths += sorted(glob.glob(os.path.join(base_dir, "scenarios", "*.json")))
    paths += sorted(glob.glob(os.path.join(base_dir, "scenarios", "generated", "*", "scenario.json")))
    return paths


def _load_demo_scenario(path: str):
    """Load a scenario config for demo mode, returning None on failure."""
    try:
        cfg = load_scenario(path)
        cfg.simulation.headless = False
        return cfg
    except Exception as exc:
        print(f"[demo] Skipping {path}: {exc}")
        return None


def _parse_args():
    parser = argparse.ArgumentParser(description='Wargames 2D simulation')
    parser.add_argument(
        '--scenario', default='scenarios/default.json',
        metavar='FILE',
        help='Scenario JSON file to load (default: scenarios/default.json)',
    )
    parser.add_argument(
        '--headless', action='store_true',
        help='Run without display; overrides scenario headless setting',
    )
    parser.add_argument(
        '--output', metavar='FILE',
        help='Write results JSON to this file; overrides scenario output.file',
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Cycle through all scenarios automatically (ignores --scenario)',
    )
    parser.add_argument(
        '--demo-timeout', type=float, default=60.0, metavar='SECONDS',
        help='Max seconds per scenario in demo mode before advancing (default: 60)',
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    # --- Demo mode: discover and load first scenario ---
    demo_mode  = args.demo
    demo_paths = discover_scenarios(_BASE_DIR) if demo_mode else []
    demo_index = 0

    if demo_mode:
        if not demo_paths:
            print("No scenarios found for demo mode.")
            sys.exit(1)
        # Find first loadable scenario
        config = None
        for i, p in enumerate(demo_paths):
            config = _load_demo_scenario(p)
            if config is not None:
                demo_index = i
                break
        if config is None:
            print("No valid scenarios found for demo mode.")
            sys.exit(1)
    else:
        config = load_scenario(args.scenario)
        # CLI flags override scenario file settings
        if args.headless:
            config.simulation.headless = True
        if args.output:
            config.simulation.output.file = args.output

    # --- Headless mode: run and exit (not available in demo mode) ---
    if not demo_mode and config.simulation.headless:
        result = run_scenario(config)
        outcome = result['outcome']
        winner  = outcome['winner']
        label   = f"Faction {winner}" if winner is not None else "Draw"
        print(f"{config.name}: {label} after {outcome['steps']} steps "
              f"({outcome['simulated_seconds']:.1f}s simulated)")
        for fid, counts in result['factions'].items():
            print(f"  Faction {fid}: {counts['remaining']}/{counts['initial']} remaining")
        sys.exit(0)

    # --- Display mode ---
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption(f"Wargames 2D — {config.name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 20)

    cam = pygame.math.Vector2(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2)
    cam_speed = 300.0
    zoom = ZOOM_DEFAULT

    state              = setup_scenario(config)
    obstacles          = state['obstacles']
    faction_objectives = state['faction_objectives']
    faction_initial_hp = {fid: sum(s.hp for s in ships)
                          for fid, ships in ShipRegistry.ships_by_faction.items()}

    selected_ship = None
    smoothed_weights: dict = {}   # class_name → smoothed float

    # Demo state
    demo_max_steps = int(args.demo_timeout * FPS)  # convert seconds → frames
    demo_step    = 0      # frame counter for current scenario
    demo_ended   = False  # True once the battle is over (counting down to next)
    demo_end_ms  = 0      # pygame.time.get_ticks() when demo_ended became True

    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.05)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            elif event.type == pygame.MOUSEWHEEL:
                factor = 1.1 ** event.y
                zoom = max(ZOOM_MIN, min(ZOOM_MAX, zoom * factor))
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mx, my = event.pos
                world_pos = screen_to_world(mx, my, cam, zoom)
                # Find closest alive ship within 20px of click
                best = None
                best_dist = 20.0
                for s in ShipRegistry.ships:
                    d = (s.position - world_pos).length()
                    if d < best_dist:
                        best_dist = d
                        best = s
                if best is not selected_ship:
                    smoothed_weights.clear()
                selected_ship = best

        # WASD camera pan (pan speed scaled so screen motion is consistent across zoom levels)
        keys = pygame.key.get_pressed()
        pan = cam_speed * dt / zoom
        if keys[pygame.K_a]:
            cam.x += pan
        if keys[pygame.K_d]:
            cam.x -= pan
        if keys[pygame.K_w]:
            cam.y += pan
        if keys[pygame.K_s]:
            cam.y -= pan

        # +/- zoom
        if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
            zoom = min(ZOOM_MAX, zoom * ZOOM_SPEED ** dt)
        if keys[pygame.K_MINUS]:
            zoom = max(ZOOM_MIN, zoom / ZOOM_SPEED ** dt)

        # Deselect if selected ship died
        if selected_ship is not None and not selected_ship.is_alive():
            selected_ship = None
            smoothed_weights.clear()

        # --- Update ---
        # Snapshot before updates so mid-frame deaths don't skip entries
        snapshot = list(ShipRegistry.ships)

        for ship in snapshot:
            if ship.is_alive() and ship.brain:
                ship.brain.update(ship, dt, ShipRegistry)

        for ship in snapshot:
            if ship.is_alive() and ship.weapon:
                ship.weapon.update(ship, dt, ShipRegistry)

        for ship in ShipRegistry.ships:
            ship.update_target(ShipRegistry)

        for ship in snapshot:
            if ship.is_alive():
                ship.update(dt)

        # Demo battle-end detection
        if demo_mode and not demo_ended:
            alive = [fid for fid, ships in ShipRegistry.ships_by_faction.items() if ships]
            cap = config.simulation.max_steps or demo_max_steps
            if len(alive) <= 1 or demo_step >= cap:
                demo_ended = True
                demo_end_ms = pygame.time.get_ticks()
            demo_step += 1

        # --- Render ---
        screen.fill((10, 10, 20))

        # Objective patrol rings (dim faction color, drawn behind everything)
        for fid, obj in faction_objectives.items():
            if not obj.is_valid():
                continue
            base = get_color(fid)
            dim = tuple(int(c * 0.30) for c in base)
            sp = world_to_screen(obj.position, cam, zoom)
            r = max(1, int(obj.patrol_radius * zoom))
            pygame.draw.circle(screen, dim, sp, r, 1)       # patrol ring
            pygame.draw.circle(screen, dim, sp, max(2, int(4 * zoom)))   # center dot

        # Obstacles (drawn first, behind everything)
        for obs in obstacles:
            sp = world_to_screen(obs.position, cam, zoom)
            r = max(int(obs.size * zoom), 2)
            pygame.draw.circle(screen, Obstacle.COLOR, sp, r)
            pygame.draw.circle(screen, Obstacle.OUTLINE_COLOR, sp, r, 2)

        # Tracers
        for ship in list(ShipRegistry.ships):
            if ship.weapon:
                for start, end, _ in ship.weapon.tracers:
                    pygame.draw.line(
                        screen, (255, 220, 80),
                        world_to_screen(start, cam, zoom),
                        world_to_screen(end, cam, zoom),
                        1,
                    )

        # Ships
        for ship in list(ShipRegistry.ships):
            color = get_color(ship.faction)
            sp = world_to_screen(ship.position, cam, zoom)
            r = max(int(ship.size * zoom), 3)

            # Selection ring (drawn beneath ship circle)
            if ship is selected_ship:
                pygame.draw.circle(screen, (255, 255, 255), sp, r + 5, 2)

            pygame.draw.circle(screen, color, sp, r)

            # Heading indicator
            heading_end = ship.position + ship.get_heading_vector() * ship.size * 2.2
            pygame.draw.line(screen, (255, 255, 255), sp, world_to_screen(heading_end, cam, zoom), 1)

            # HP bar
            bar_w = r * 3
            bar_h = 3
            bx = sp[0] - bar_w // 2
            by = sp[1] - r - 6
            hp_frac = max(ship.hp / max(ship.max_hp, 1e-6), 0.0)
            pygame.draw.rect(screen, (80, 0, 0), (bx, by, bar_w, bar_h))
            pygame.draw.rect(screen, (0, 200, 0), (bx, by, int(bar_w * hp_frac), bar_h))

            # Shield bar (below HP bar, only when the ship has a shield pool)
            if ship.max_shield > 0:
                shield_by = by + bar_h + 2
                sh_frac = max(0.0, ship.shield / ship.max_shield)
                pygame.draw.rect(screen, (10, 30, 80), (bx, shield_by, bar_w, bar_h))
                if sh_frac > 0:
                    pygame.draw.rect(screen, (0, 160, 220),
                                     (bx, shield_by, int(bar_w * sh_frac), bar_h))

            # Ammo segments (above HP bar)
            if ship.weapon:
                n = ship.weapon.cassette_size
                seg_w, seg_h, seg_gap = 5, 3, 1
                total_seg_w = n * seg_w + (n - 1) * seg_gap
                sx0 = sp[0] - total_seg_w // 2
                sy = by - seg_h - 2          # 2 px gap above HP bar

                if ship.weapon.reloading:
                    # Smooth fill: segments refill left-to-right as timer counts down
                    progress = 1.0 - ship.weapon.reload_timer / max(ship.weapon.reload_duration, 1e-6)
                    filled_f = progress * n
                else:
                    filled_f = float(n - ship.weapon.shots_fired)

                for i in range(n):
                    sx = sx0 + i * (seg_w + seg_gap)
                    if i < int(filled_f):
                        # Fully filled
                        pygame.draw.rect(screen, color, (sx, sy, seg_w, seg_h))
                    elif i == int(filled_f) and (filled_f % 1) > 1e-4:
                        # Partially filled — only during reload animation
                        frac = filled_f - int(filled_f)
                        pygame.draw.rect(screen, (40, 40, 40), (sx, sy, seg_w, seg_h))
                        pygame.draw.rect(screen, color, (sx, sy, max(1, int(seg_w * frac)), seg_h))
                    else:
                        # Empty
                        pygame.draw.rect(screen, (40, 40, 40), (sx, sy, seg_w, seg_h))

        # HUD
        hud_y = 10
        if demo_mode:
            name_surf = font.render(config.name, True, (200, 200, 100))
            screen.blit(name_surf, (WINDOW_WIDTH // 2 - name_surf.get_width() // 2, hud_y))
            hud_y += 18

        n0 = len(ShipRegistry.ships_by_faction.get(0, []))
        n1 = len(ShipRegistry.ships_by_faction.get(1, []))
        total = len(ShipRegistry.ships)
        sel_hint = "  click=select" if selected_ship is None else "  click=deselect"
        hud = font.render(
            f"Green: {n0}   Red: {n1}   Total: {total}   WASD=pan  +/-=zoom({zoom:.2f}x)  scroll=zoom  ESC=quit{sel_hint}",
            True, (200, 200, 200),
        )
        screen.blit(hud, (10, hud_y))

        # Team progress bars
        _draw_team_bars(screen, font, faction_initial_hp)

        # Weight overlay (only when a ship is selected)
        if selected_ship is not None and selected_ship.is_alive():
            _draw_weight_overlay(screen, font, selected_ship, smoothed_weights, dt)

        # Demo transition overlay
        if demo_mode and demo_ended:
            elapsed_ms  = pygame.time.get_ticks() - demo_end_ms
            remaining_ms = DEMO_TRANSITION_MS - elapsed_ms

            if remaining_ms <= 0:
                # Advance to next working scenario
                next_config = None
                for _ in range(len(demo_paths)):
                    demo_index = (demo_index + 1) % len(demo_paths)
                    next_config = _load_demo_scenario(demo_paths[demo_index])
                    if next_config is not None:
                        break
                if next_config is None:
                    print("[demo] All scenarios failed to load — exiting.")
                    running = False
                else:
                    config = next_config
                    state  = setup_scenario(config)
                    obstacles          = state['obstacles']
                    faction_objectives = state['faction_objectives']
                    faction_initial_hp = {fid: sum(s.hp for s in ships)
                                          for fid, ships in ShipRegistry.ships_by_faction.items()}
                    cam.x, cam.y = WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2
                    zoom = ZOOM_DEFAULT
                    selected_ship = None
                    smoothed_weights.clear()
                    demo_step   = 0
                    demo_ended  = False
                    pygame.display.set_caption(f"Wargames 2D — {config.name}")
            else:
                secs_left = (remaining_ms // 1000) + 1
                overlay = pygame.Surface((WINDOW_WIDTH, 80), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 180))
                screen.blit(overlay, (0, WINDOW_HEIGHT // 2 - 40))

                alive = [fid for fid, ships in ShipRegistry.ships_by_faction.items() if ships]
                result_text = f"Faction {alive[0]} wins!" if alive else "Draw"
                next_path = demo_paths[(demo_index + 1) % len(demo_paths)]
                next_name = os.path.basename(os.path.dirname(next_path)) or os.path.basename(next_path)

                line1 = font.render(f"{result_text}  —  Next in {secs_left}s", True, (255, 255, 255))
                line2 = font.render(next_name, True, (180, 180, 180))
                screen.blit(line1, (WINDOW_WIDTH // 2 - line1.get_width() // 2, WINDOW_HEIGHT // 2 - 25))
                screen.blit(line2, (WINDOW_WIDTH // 2 - line2.get_width() // 2, WINDOW_HEIGHT // 2 + 5))

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
