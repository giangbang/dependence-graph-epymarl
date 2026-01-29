import json
import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

import custom_smaclite.env.units.targeters.targeter as t
from smaclite.env.units.combat_type import CombatType
from smaclite.env.util.plane import Plane

from smaclite.env.units.unit_type import Attribute, MELEE_ATTACK_RANGE

TARGETER_CACHE: Dict[str, t.Targeter] = {}

@dataclass
class UnitStats(object):
    name: str
    hp: int
    armor: int
    damage: int
    cooldown: float
    speed: float
    attack_range: int
    size: float
    attributes: Set[Attribute]
    valid_targets: Set[Plane]
    shield: int = 0
    energy: int = 0
    starting_energy: int = 0
    attacks: int = 1
    combat_type: CombatType = CombatType.DAMAGE
    minimum_scan_range: int = 5
    bonuses: Dict[Attribute, float] = None
    plane: Plane = Plane.GROUND
    hp_regen: float = 0

    @classmethod
    def from_file(cls, filename, custom_unit_path):
        if not os.path.isabs(filename):
            filename = os.path.join(os.path.abspath(custom_unit_path),
                                    filename)
        if not filename.endswith(".json"):
            filename += ".json"
        with open(filename) as f:
            stats_dict = json.load(f)
        stats_dict['name'] = os.path.splitext(os.path.basename(filename))[0]
        if stats_dict['attack_range'] == "MELEE":
            stats_dict['attack_range'] = MELEE_ATTACK_RANGE
        stats_dict['attributes'] = set(map(Attribute,
                                           stats_dict['attributes']))
        stats_dict['valid_targets'] = set(map(Plane,
                                              stats_dict['valid_targets']))
        if 'bonuses' in stats_dict:
            stats_dict['bonuses'] = {Attribute(k): v for k, v
                                     in stats_dict['bonuses'].items()}
        if 'combat_type' in stats_dict:
            stats_dict['combat_type'] = CombatType(stats_dict['combat_type'])
        if 'plane' in stats_dict:
            stats_dict['plane'] = Plane(stats_dict['plane'])
        targeter_kwargs = stats_dict.pop('targeter_kwargs', {})
        TARGETER_CACHE[stats_dict['name']] = \
            t.TargeterType[stats_dict.pop(
                'targeter', 'STANDARD')].value(**targeter_kwargs)
        return cls(**stats_dict)