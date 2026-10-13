import math
import random
import pygame

from entities.ship import Ship
from entities.weapon import Weapon
from behavior.ship_brain import ShipBrain
from behavior.personality import DefaultPersonality
from behavior.navigate.objective_module import ObjectiveModule
from behavior.navigate.bounds_module import BoundsModule
from behavior.navigate.collision_avoidance_module import CollisionAvoidanceModule
from behavior.navigate.separation_module import SeparationModule
from behavior.flock.alignment_module import AlignmentModule
from behavior.flock.cohesion_module import CohesionModule
from behavior.engage.evasion_module import EvasionModule
from behavior.engage.firing_solution_module import FiringSolutionModule


def _make_brain(personality=None) -> ShipBrain:
    return ShipBrain(
        modules=[
            ObjectiveModule(inner_radius=80.0),
            BoundsModule(bounds_radius=2000.0),
            CollisionAvoidanceModule(neighbor_radius=100.0),
            SeparationModule(neighbor_radius=80.0),
            AlignmentModule(neighbor_radius=200.0),
            CohesionModule(neighbor_radius=200.0),
            EvasionModule(detection_range=600.0),
            FiringSolutionModule(detection_range=600.0),
        ],
        personality=personality if personality is not None else DefaultPersonality(),
    )


def _spawn_ship(faction, center, objective, spawn_radius, personality, ship_cfg, weapon_cfg):
    angle = random.uniform(0, math.tau)
    r = random.uniform(0, spawn_radius)
    pos = center + pygame.math.Vector2(math.cos(angle) * r, math.sin(angle) * r)

    size = ship_cfg.size if ship_cfg else 10.0
    ship = Ship(pos, faction=faction, size=size)
    ship.objective = objective

    if ship_cfg:
        ship.max_speed            = ship_cfg.max_speed
        ship.forward_acceleration = ship_cfg.forward_acceleration
        ship.brake_acceleration   = ship_cfg.brake_acceleration
        ship.turn_rate            = ship_cfg.turn_rate
        ship.lateral_damping      = ship_cfg.lateral_damping
        ship.hp                   = ship_cfg.hp
        ship.max_hp               = ship_cfg.hp       # freeze max alongside current
        ship.shield               = ship_cfg.shield
        ship.max_shield           = ship_cfg.shield   # max = initial
        ship.shield_regen         = ship_cfg.shield_regen
        ship.visual_range         = ship_cfg.visual_range

    ship.heading_angle = random.uniform(0, 360)

    weapon = Weapon()
    if weapon_cfg:
        weapon.range           = weapon_cfg.range
        weapon.damage          = weapon_cfg.damage
        weapon.fire_rate       = weapon_cfg.fire_rate
        weapon.max_fire_angle  = weapon_cfg.max_fire_angle
        weapon.cassette_size   = weapon_cfg.cassette_size
        weapon.reload_duration = weapon_cfg.reload_duration
    weapon.initialize(ship)
    ship.weapon = weapon

    ship.brain = _make_brain(personality)
    return ship


def spawn_faction(
    faction: int,
    center: pygame.math.Vector2,
    objective,                          # Objective or Vector2
    groups: list,                       # list[ShipGroupConfig]
    spawn_radius: float = 150.0,
    personality=None,                   # Personality instance, shared by all ships
) -> list:
    ships = []
    for group in groups:
        for _ in range(group.ship_count):
            ship = _spawn_ship(faction, center, objective, spawn_radius, personality,
                               group.ship, group.weapon)
            ships.append(ship)
    return ships
