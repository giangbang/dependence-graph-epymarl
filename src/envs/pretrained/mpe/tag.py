from pathlib import Path

import gymnasium as gym
from gymnasium.spaces import Tuple
import numpy as np
import torch

from .ddpg import DDPG


class FrozenTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pettingzoo.mpe.simple_tag.simple_tag import raw_env
        env: raw_env = self.env.unwrapped._env.unwrapped

        self.tot_agents = len(self.env.action_space)
        self.n_agents = len(env.scenario.adversaries(env.world))
        self.n_pt_agents = self.tot_agents - self.n_agents

        self.pt_action_space = self.action_space[-self.n_pt_agents:]
        self.pt_observation_space = self.observation_space[-self.n_pt_agents:]

        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])
        self.unwrapped.n_agents = self.n_agents

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        return obss[:self.n_agents], info

    def step(self, action):
        random_action = 0
        action = tuple(action) + (random_action,)
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]
        return obs, rew, done, truncated, info


class RandomTag(gym.Wrapper):
    """Tag with pretrained prey agent"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pettingzoo.mpe.simple_tag.simple_tag import raw_env
        env: raw_env = self.env.unwrapped._env.unwrapped

        self.tot_agents = len(self.env.action_space)
        self.n_agents = len(env.scenario.adversaries(env.world))
        self.n_pt_agents = self.tot_agents - self.n_agents

        self.pt_action_space = self.action_space[-self.n_pt_agents:]
        self.pt_observation_space = self.observation_space[-self.n_pt_agents:]

        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])
        self.unwrapped.n_agents = self.n_agents

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        return obss[:self.n_agents], info

    def step(self, action):
        random_action = (pt_action_space.sample() for pt_action_space in self.pt_action_space)
        action = tuple(action) + random_action
        obs, rew, done, truncated, info = super().step(action)
        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]
        return obs, rew, done, truncated, info


class PretrainedTag(gym.Wrapper):
    """Tag with pretrained prey agent"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from pettingzoo.mpe.simple_tag.simple_tag import raw_env
        env: raw_env = self.env.unwrapped._env.unwrapped

        self.tot_agents = len(self.env.action_space)
        self.n_agents = len(env.scenario.adversaries(env.world))
        self.n_pt_agents = self.tot_agents - self.n_agents
        if self.n_agents != 3 or self.n_pt_agents != 1:
            raise NotImplementedError("Pretrained Tag only works on default configuration!")

        self.pt_action_space = self.action_space[-self.n_pt_agents:]
        self.pt_observation_space = self.observation_space[-self.n_pt_agents:]

        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])
        self.unwrapped.n_agents = self.n_agents

        self.prey = DDPG(14, 5, 50, 128, 0.01)  # 14 is the observation shape of tag
        # current file dir
        param_path = Path(__file__).parent / "prey_params.pt"
        save_dict = torch.load(param_path)
        self.prey.load_params(save_dict["agent_params"][-1])
        self.prey.policy.eval()
        self.last_prey_obs = None

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_prey_obs = obs[self.n_agents:]
        return obs[:self.n_agents], info

    def step(self, action):
        prey_action = self.prey.step(self.last_prey_obs)
        action = np.concatenate([action, prey_action])
        obs, rew, done, truncated, info = super().step(action)
        self.last_prey_obs = obs[self.n_agents:]

        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]
        return obs, rew, done, truncated, info
