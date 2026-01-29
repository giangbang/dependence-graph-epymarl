from collections.abc import Iterable
import warnings

import gymnasium as gym
from gymnasium.spaces import flatdim
from gymnasium.wrappers import TimeLimit
import envs.custom_smaclite  # noqa

from .multiagentenv import MultiAgentEnv


class SMACliteWrapper(MultiAgentEnv):

    def __init__(
        self,
        map_name,
        seed,
        time_limit,
        common_reward=True,  # ignored in smac/smaclite
        reward_scalarisation="sum",  # ignored in smac/smaclite
        **kwargs,
    ):
        # initiate `smaclite/{}-v0` or `custom-smaclite/{}-v0`
        self.env = gym.make(f"{map_name}-v0", seed=seed, **kwargs)
        self.env = TimeLimit(self.env, max_episode_steps=time_limit)

        self.n_agents = self.env.unwrapped.n_agents
        self.episode_limit = time_limit

        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)
        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def step(self, actions):
        """Returns obss, reward, terminated, truncated, info"""
        actions = [int(act) for act in actions]
        obs, reward, terminated, truncated, info = self.env.step(actions)

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        return obs, reward, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self.env.unwrapped.get_obs()

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self.env.unwrapped.get_obs()[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        return self.env.unwrapped.obs_size

    def get_state(self):
        return self.env.unwrapped.get_state()

    def get_state_size(self):
        """Returns the shape of the state"""
        return self.env.unwrapped.state_size

    def get_avail_actions(self):
        return self.env.unwrapped.get_avail_actions()

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return self.env.unwrapped.get_avail_actions()[agent_id]

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return flatdim(self.longest_action_space)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs = self.env.reset(seed=seed, options=options)
        return obs, {}

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)
