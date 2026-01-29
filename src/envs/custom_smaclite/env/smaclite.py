from typing import Dict, List, Tuple

import gymnasium as gym
from gymnasium.utils import seeding
import numpy as np

from smaclite.env.smaclite import SMACliteEnv
from smaclite.env.units.unit_type import CombatType, UnitType
from smaclite.env.maps.map import Group
from smaclite.env.util.faction import Faction


from smaclite.env.smaclite import STEP_MUL, GROUP_BUFFER

from custom_smaclite.env.units.unit import Unit


class CustomSMACliteEnv(SMACliteEnv):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.use_targeter_graph = kwargs.get("use_targeter_graph", False)

    def step(self, actions):
        assert len(actions) == self.n_agents
        assert all(type(action) == int for action in actions)
        self.last_actions = np.eye(self.n_actions)[np.array(actions)].flatten()
        avail_actions = self.get_avail_actions()
        for i, action in enumerate(actions):
            if i not in self.agents:
                assert actions[i] == 0
                continue
            agent = self.agents[i]
            if not avail_actions[i][action]:
                raise ValueError(f"Invalid action for agent {i}: {action}")
            agent.command = self._SMACliteEnv__get_command(agent, action)
        rewards_simsteps = tuple(
            self.__world_step()[None, ...] for _ in range(STEP_MUL)
        )

        rewards = np.concatenate(rewards_simsteps, axis=0).sum(axis=0)
        all_enemies_dead = len(self.enemies) == 0
        if all_enemies_dead:
            rewards += 200 / self.n_agents  # distribute team rewards for all agents
        done = all_enemies_dead or len(self.agents) == 0
        truncated = False
        rewards /= self.max_reward / 20  # Scale reward between 0 and 20
        rewards = rewards.tolist()  # for compatibility with epymarl
        return self.get_obs(), rewards, done, truncated, self._SMACliteEnv__get_info()

    def __world_step(self):
        # Do not automatically render the environment unless render() is called
        # if self.renderer is not None:
        #     self.render()
        for unit in self.all_units.values():
            unit.clean_up_target()
        # NOTE There is an assumption here that the set of attack-moving units
        # will never include any allied units. This is true right now,
        # but might change in the future.
        if attackmoving_units := [
            enemy
            for enemy in self.enemies.values()
            if enemy.combat_type == CombatType.DAMAGE and enemy.target is None
        ]:
            attackmoving_radii = [
                unit.minimum_scan_range + self.max_unit_radius
                for unit in attackmoving_units
            ]
            attackmoving_targets = self.neighbour_finder_ally.query_radius(
                attackmoving_units,
                attackmoving_radii,
                return_distance=True,
                targetting_mode=True,
            )
            for unit, targets in zip(attackmoving_units, attackmoving_targets):
                unit.potential_targets = targets
            for unit in self.agents.values():
                if unit.target is not None and unit.plane in unit.target.valid_targets:
                    unit.target.potential_targets.append((unit, 2e9))
        if healmoving_units := [
            enemy
            for enemy in self.enemies.values()
            if enemy.combat_type == CombatType.HEALING and enemy.target is None
        ]:
            healmoving_radii = [
                unit.minimum_scan_range + self.max_unit_radius
                for unit in healmoving_units
            ]
            attackhealing_targets = self.neighbour_finder_enemy.query_radius(
                healmoving_units,
                healmoving_radii,
                return_distance=True,
                targetting_mode=True,
            )
            for unit, targets in zip(healmoving_units, attackhealing_targets):
                unit.potential_targets = targets
        if any(
            unit.combat_type == CombatType.HEALING for unit in self.agents.values()
        ) and (
            nonpriority_attackmoving := [
                enemy
                for enemy in self.enemies.values()
                if enemy.combat_type == CombatType.DAMAGE
                and enemy.target is not None
                and enemy.target.combat_type != CombatType.HEALING
            ]
        ):
            attackmoving_radii = [
                unit.minimum_scan_range + self.max_unit_radius
                for unit in nonpriority_attackmoving
            ]
            attackmoving_targets = self.neighbour_finder_ally.query_radius(
                nonpriority_attackmoving,
                attackmoving_radii,
                return_distance=True,
                targetting_mode=True,
            )
            for unit, targets in zip(nonpriority_attackmoving, attackmoving_targets):
                unit.priority_targets = targets
        for unit in self.all_units.values():
            unit.prepare_velocity()
        self.velocity_updater.compute_new_velocities(self.all_units)

        # only contains alive units
        units = list(self.all_units.values())
        units_id = list(self.all_units.keys())

        shuffled_id = np.arange(len(units))  # shuffle alive units
        self._np_random.shuffle(shuffled_id)

        # setup graph agent
        if self.use_targeter_graph:
            enemy_graph, ally_graph = {}, {}
        else:
            enemy_graph, ally_graph = None, None
        shuffled_rewards = tuple(
            units[id].game_step(
                neighbour_finder=self._SMACliteEnv__get_targeter_neighbour_finder(
                    units[id]
                ),
                max_radius=self.max_unit_radius,
                enemy_graph=enemy_graph,
                ally_graph=ally_graph,
            )
            for id in shuffled_id
        )

        rewards = np.zeros((self.n_agents,), dtype=np.float32)

        # contains ally agents, exclude dead ones
        alive_agents = set(self.agents.keys())

        for id, rw in zip(shuffled_id, shuffled_rewards):
            unshuffled_id = units_id[id]  # id of unshuffled units, include the deads
            if unshuffled_id in alive_agents:
                rewards[unshuffled_id] = rw

        # if enemies damage other enemies, rewards are distributed to all agents
        remain_rewards = np.sum(shuffled_rewards) - np.sum(rewards)
        remain_rewards = max(0, remain_rewards)
        rewards += remain_rewards / self.n_agents

        # update graph agent
        if self.use_targeter_graph:
            self.graph = np.diag(np.ones(self.n_agents, dtype=bool))
            for enemy, connected_allies in enemy_graph.items():
                self.graph[connected_allies, connected_allies] = 1
            for ally, connected_allies in ally_graph.items():
                self.graph[connected_allies, ally] = 1

        self._SMACliteEnv__update_deaths()
        self.neighbour_finder_all.update()
        self.neighbour_finder_ally.update()
        self.neighbour_finder_enemy.update()
        return rewards

    def reset(self, **kwargs):
        if self.use_targeter_graph:
            self.graph = np.diag(np.ones(self.n_agents))
        return super().reset(**kwargs)

    def __place_group(self, group: Group):
        faction = group.faction
        faction_dict = self.agents if faction == Faction.ALLY else self.enemies
        all_types_in_group: List[UnitType] = []
        for unit_type, count in group.units:
            all_types_in_group.extend([unit_type] * count)
        group_size = len(all_types_in_group)
        square_side = np.ceil(np.sqrt(group_size)).astype(int)
        unit_grid = [[None for _ in range(square_side)] for _ in range(square_side)]
        a = b = 0
        # Plan out the layout of the units in the group
        for unit_type in all_types_in_group:
            unit_grid[b][a] = unit_type
            a += 1
            if a == square_side:
                a = 0
                b += 1
        row_radii = [max(u.radius if u else 0 for u in row) for row in unit_grid]
        prev_row_height = 0
        group_height = 2 * sum(row_radii) + (square_side - 1) * GROUP_BUFFER
        row_widths = [sum(u.size if u else 0 for u in row) for row in unit_grid]
        group_width = max(row_widths)
        # This is so enemy units spawn opposite allied units
        # i.e. the layout is center-symmetric if the groups are equal
        m = 1 if faction == Faction.ALLY else -1
        x0, y = group.x - m * group_width / 2, group.y - m * group_height / 2
        # Actually place the units
        for i, row in enumerate(unit_grid):
            x = x0
            y += m * (prev_row_height + row_radii[i])
            prev_row_height = row_radii[i]
            prev_unit_width = 0
            for u in row:
                if u is None:
                    continue
                x += m * (prev_unit_width + u.radius)
                prev_unit_width = u.radius
                id_overall = len(self.all_units)
                id_in_faction = len(faction_dict)
                unit = Unit(u, faction, x, y, id_overall, id_in_faction)
                # Uncomment to test killing units
                # unit.hp = np.random.choice([0, np.random.randint(unit.hp)])
                self.all_units[id_overall] = unit
                faction_dict[id_in_faction] = unit
                x += m * GROUP_BUFFER
            y += m * GROUP_BUFFER
