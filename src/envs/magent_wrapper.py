from collections.abc import Iterable
import copy
import importlib
from pathlib import Path
from gymnasium.spaces import Tuple
import warnings
import magent2
import magent2.environments
import magent2.environments.magent_env
import numpy as np
from magent2 import GridWorld

from .pz_wrapper import PettingZooWrapper
from .multiagentenv import MultiAgentEnv
import gymnasium as gym
from gymnasium.spaces import flatdim

from magent2.environments.magent_env import magent_parallel_env


def remove_null_items(input_dict):
    rm_keys = []
    for key, val in input_dict.items():
        if val is None:
            rm_keys.append(key)
    for key in rm_keys:
        input_dict.pop(key)
    return input_dict

class MAgentEnv(PettingZooWrapper):
    def __init__(self, map_name, **kwargs):
        env_name = map_name
        env = importlib.import_module(f"magent2.environments.{env_name}")
        self.env_name = env_name
        kwargs["render_mode"] = "rgb_array"

        kwargs = remove_null_items(kwargs)

        self._env: magent_parallel_env = env.parallel_env(**kwargs)
        self.n_agents = len(self._env.possible_agents)
        self.agents = copy.deepcopy(self._env.agents)
        self.last_obs = None
        self.core_magent_world: GridWorld = self._env.unwrapped.env
        self.episode_limit = self._env.max_cycles

        self.action_space = Tuple(
            tuple([self._env.action_spaces[k] for k in self._env.agents])
        )
        self.observation_space = Tuple(
            tuple([self._env.observation_spaces[k] for k in self._env.agents])
        )

        # all agents, including the deads
        self.possible_agents = self._env.possible_agents[:]

    def render(self, **kwargs):
        rgb = self._env.render(**kwargs)

        return rgb

    def _revert_pos_and_unflatten(self, obs, obs_shape):
        obs = obs[..., :-2]
        obs = obs.view(-1, *obs_shape)
        return obs

    def construct_pos_vec(self, all_pos):
        vec = np.zeros((self.n_agents, 2))
        if len(self._env.agents) == 0: 
            return vec
        assert len(all_pos) == len(self._env.agents), f"{len(all_pos)} {len(self._env.agents)}"
        j = 0
        for i, agent in enumerate(self._env.possible_agents):
            if agent in self._env.agents:
                vec[i] = all_pos[j]
                j += 1
        return vec / 10


    def reset(self, *args, **kwargs):
        obs, info = self._env.reset(*args, **kwargs)
        # obs = self._append_pos(obs)
        obs = tuple([obs[k] for k in self._env.agents])

        self.last_obs = obs
        return obs, info

    def get_pos(self):
        pos_vec = []
        for handle in self._env.handles:
            pos = self._env.env.get_pos(handle)  # n_agent_in_handle * 2
            pos_vec.append(pos)
        return np.concatenate(pos_vec, axis=0)

    def step(self, actions):
        dict_actions = {}
        assert len(actions) == len(
            self.possible_agents
        ), f"{len(actions)} {len(self.possible_agents)}"

        for agent, action in zip(self.possible_agents, actions):
            dict_actions[agent] = action

        observations, rewards, dones, truncated, infos = self._env.step(dict_actions)

        # observations = self._append_pos(observations)

        observations_vec = [None] * self.n_agents
        rewards_vec = np.zeros((self.n_agents), np.float32)
        for i, agent in enumerate(self._env.possible_agents):
            if agent in observations:
                observations_vec[i] = observations[agent]
            else:
                observations_vec[i] = self._env._zero_obs[agent]
            if agent in rewards:
                rewards_vec[i] = rewards[agent]

        obs = observations_vec
        rewards = rewards_vec

        done = all([dones[k] for k in self._env.agents])
        truncated = all([truncated[k] for k in self._env.agents])
        info = {
            f"{k}_{key}": value
            for k in self._env.agents
            for key, value in infos[k].items()
        }
        assert len(obs) == self.n_agents, f"{len(obs)} != {self.n_agents}"
        self.last_obs = obs
        return obs, rewards, done, truncated, info


# import all files within the magent2 library that match environments/*_v?.py"
envs = Path(magent2.__path__[0]).glob("environments/*_v?.py")
for e in envs:
    name = e.stem.replace("_", "-")
    filename = e.stem

    gymkey = f"magent2/{name}"
    gym.register(
        gymkey,
        entry_point="envs.magent_wrapper:MAgentEnv",
        kwargs={
            "map_name": filename,
        },
    )


class MAgentWrapper(MultiAgentEnv):

    def __init__(
        self,
        seed,
        map_name,
        pretrained_wrapper: bool,
        common_reward,
        reward_scalarisation,
        **kwargs,
    ):
        env_name = map_name
        self._env: MAgentEnv = gym.make(env_name, **kwargs)
        self.env_name = env_name
        self.episode_limit = self._env.unwrapped.episode_limit
        self.pretrained_wrapper = pretrained_wrapper

        if pretrained_wrapper:
            import envs.pretrained.magent as pretrained  # noqa

            if "battle" in map_name:
                self._env = pretrained.Battle(self._env)
            elif "adversarial-pursuit" in map_name:
                self._env = pretrained.Pursuit(self._env)
            else:
                raise NotImplemented

        self.n_agents = len(self._env.action_space)
        self.n_tot_agent = len(self._env.unwrapped.action_space)
        self.magent_parallel_env: (
            magent2.environments.magent_env.magent_parallel_env
        ) = self._env.unwrapped._env

        self._obs = None
        self._info = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = seed
        try:
            self._env.unwrapped.seed(self._seed)
        except:
            self._env.reset(seed=self._seed)

        self.common_reward = common_reward
        self.state_size = None
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = lambda rewards: sum(rewards)
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = lambda rewards: sum(rewards) / len(rewards)
            else:
                raise ValueError(
                    f"Invalid reward_scalarisation: {reward_scalarisation} (only support 'sum' or 'mean')"
                )

    def _pad_observation(self, obs):
        padded_obs = []
        for i, o in enumerate(obs):
            padwidth = np.expand_dims(
                np.array(self.longest_observation_space.shape)
                - np.array(self._env.observation_space[i].shape),
                axis=-1,
            )
            padwidth = np.concatenate((np.zeros_like(padwidth), padwidth), axis=-1)
            pad = np.pad(
                o,
                padwidth,
                "constant",
                constant_values=0,
            )
            padded_obs.append(pad)
        return padded_obs

    def step(self, action):
        """Returns obss, reward, terminated, truncated, info"""
        self._obs, reward, done, truncated, self._info = self._env.step(action)
        self._obs = self._pad_observation(self._obs)

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        shape = self.longest_observation_space.shape
        if len(shape) == 1:
            shape = shape[0]
        return shape

    def get_state(self):
        return self._env.unwrapped._env.state()
        

    def get_state_size(self):
        """Returns the shape of the state"""
        return self._env.unwrapped._env.base_state.shape

    def get_avail_actions(self):
        avail_actions = []
        # magent_world: GridWorld = self._env.unwrapped.core_magent_world
        agent_alive = np.zeros((self.n_tot_agent,), bool)
        for handle in self.magent_parallel_env.handles:
            ids = self.magent_parallel_env.env.get_agent_id(handle)
            agent_alive[ids] = 1

        for agent_id in range(self.n_tot_agent):
            if agent_alive[agent_id]:
                avail_agent = self.get_avail_agent_actions(agent_id)
            else:
                avail_agent = np.zeros(self.longest_action_space.n, dtype=int)
                avail_agent[0] = 1
            avail_actions.append(avail_agent)
        return list(avail_actions[: self.n_agents])

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id, not consider death status"""
        valid = flatdim(self._env.unwrapped.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        assert self.longest_action_space.n == len(valid) + len(invalid), f"{self.longest_action_space.n} {len(valid)}"
        return valid + invalid

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """Returns initial observations and info"""
        if not self.state_size:
            self.state_size = self.get_state_size()
        self._obs, self._info = self._env.reset()
        self._obs = self._pad_observation(self._obs)
        return self._obs, self._info

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        self._env.close()

    def seed(self, seed=None):
        self._env.unwrapped.seed(seed)

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

if __name__ == "__main__":
    env  = MAgentWrapper(1,
            "magent2/battle-v4",
            None,
            False,
            "sum",
            render_mode="rgb_array",)
    obs, _ = env.reset()
    print(obs[0].shape)
    obs, rewards, done, truncated, info = env.step([0]*162)
    print(obs[0].shape)
    print(env._env.observation_space)
    print(env.get_env_info())