from enum import Enum
from typing import Callable
from smaclite.env.util.faction import Faction

import numpy as np
from smaclite.env.util import point_inside_circle


def add_to_graph(origin, target, **kwargs):
    if "enemy_graph" not in kwargs or kwargs["enemy_graph"] is None:
        return
    if origin.faction == Faction.ALLY:
        if target.faction == Faction.ENEMY:
            enemy_graph: dict[int, set[int]] = kwargs["enemy_graph"]
            enemy_graph[target.id] = enemy_graph.get(target.id, set()) | {origin.id}
        else:
            ally_graph: dict[int, set[int]] = kwargs["ally_graph"]
            ally_graph[target.id] = ally_graph.get(target.id, set()) | {origin.id}


class Targeter(object):
    def target(self, origin, target, **kwargs) -> float:
        raise NotImplementedError


class StandardTargeter(Targeter):
    def target(self, origin, target, **kwargs) -> float:
        add_to_graph(origin, target, **kwargs)
        return origin.deal_damage(target)


class KamikazeTargeter(Targeter):
    """A type of targeter that explodes in a radius around it upon attacking,
    then dies.
    """

    def __init__(self, radius: float) -> None:
        self.radius = radius

    def target(self, origin, target, **kwargs) -> float:
        neighbour_finder = kwargs["neighbour_finder"]
        max_radius = kwargs["max_radius"]
        reward_bonus = 0 if origin.faction == Faction.ALLY else origin.hp
        origin.hp = 0
        neighbours = neighbour_finder.query_radius([origin], self.radius + max_radius)[
            0
        ]
        res = 0
        for target in neighbours:
            if point_inside_circle(target.pos, origin.pos, self.radius + target.radius):
                res += origin.deal_damage(target)
                add_to_graph(origin, target, **kwargs)
        # return (
        #     sum(
        #         origin.deal_damage(target)
        #         for target in neighbours
        #         if point_inside_circle(
        #             target.pos, origin.pos, self.radius + target.radius
        #         )
        #     )
        #     + reward_bonus
        # )

        return res


class LaserBeamTargeter(Targeter):
    """A type of targeter that fires a laser line perpendicular to the
    line from the origin to the target, hitting all the units the laser
    line touches.
    """

    def __init__(self, width, height) -> None:
        self.width = width
        self.height = height
        self.offset = np.array([width / 2, height / 2])
        self.radius = np.hypot(self.width, self.height) / 2

    def target(self, origin, target, **kwargs) -> float:
        neighbour_finder = kwargs["neighbour_finder"]

        neighbours = neighbour_finder.query_radius([target], self.radius)[0]
        poses = np.array([neighbour.pos for neighbour in neighbours])
        transform_function = self.__get_transform_function(origin, target)
        transformed_target = transform_function(target.pos)
        transformed_poses = transform_function(poses)
        dx_dy = (np.abs(-transformed_poses + transformed_target) - self.offset).clip(
            min=0
        )
        dists_sq = (dx_dy**2).sum(axis=1)

        res = 0
        for i, unit in enumerate(neighbours):
            if dists_sq[i] <= unit.radius_sq:
                res += origin.deal_damage(unit)
                add_to_graph(origin, unit, **kwargs)
        return res

        # return sum(
        #     origin.deal_damage(unit)
        #     for i, unit in enumerate(neighbours)
        #     if dists_sq[i] <= unit.radius_sq
        # )

    def __get_transform_function(
        self, origin, target
    ) -> Callable[[np.ndarray], np.ndarray]:
        diff = target.pos - origin.pos
        theta = np.arctan(-1 / (diff[1] / diff[0]))
        c, s = np.cos(theta), np.sin(theta)
        rot_matrix = np.array([[c, -s], [s, c]])
        return lambda x: np.dot(x, rot_matrix)


class HealTargeter(Targeter):
    def target(self, origin, target, **kwargs) -> float:
        origin.heal(target)
        add_to_graph(origin, target, **kwargs)
        return 0


class TargeterType(Enum):
    STANDARD = StandardTargeter
    KAMIKAZE = KamikazeTargeter
    LASER_BEAM = LaserBeamTargeter
    HEAL = HealTargeter
