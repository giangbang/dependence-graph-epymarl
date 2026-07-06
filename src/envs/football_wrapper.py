import numpy as np
from envs.multiagentenv import MultiAgentEnv
from gym.wrappers import TimeLimit
import warnings
from collections.abc import Iterable
from gymnasium.spaces import flatdim



env_num_agents = {
    "academy_pass_and_shoot_with_keeper": 2,
    "academy_run_pass_and_shoot_with_keeper": 2,
    "academy_3_vs_1_with_keeper": 3,
    "academy_counterattack_easy": 4,
    "academy_counterattack_hard": 4,
    "academy_corner": 11,
    "academy_single_goal_versus_lazy": 11,
}

class FootballWrapper(MultiAgentEnv):
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
        import envs.custom_football as football_env
        self._env = football_env.create_environment(
            env_name=map_name, 
            write_goal_dumps=False, 
            write_full_episode_dumps=False, 
            dump_frequency=0,
            write_video=False,
            render=False,
            number_of_left_players_agent_controls=env_num_agents[map_name], 
            **kwargs
        )
        self.env = TimeLimit(self._env, max_episode_steps=time_limit)
        self._obs = None
        self.graph = None
        self.previous_designated = None

        self.n_agents = self.env.observation_space.shape[0]
        self.episode_limit = time_limit
        self.observation_space = self.env.observation_space

        self.action_space = self.env.action_space
        # e.g. MultiDiscrete([19 19 19 19])
        self.common_reward = common_reward
        if self.common_reward:
            if reward_scalarisation == "sum":
                self.reward_agg_fn = np.sum
            elif reward_scalarisation == "mean":
                self.reward_agg_fn = np.mean
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
        self._build_graph()
        obs = self.split(obs)
        self._obs = obs

        return obs, reward, terminated, truncated, info

    def get_obs(self):
        """Returns all agent observations in a list"""
        return self._obs

    def get_obs_agent(self, agent_id):
        """Returns observation for agent_id"""
        return self._obs[agent_id]

    def get_obs_size(self):
        """Returns the shape of the observation"""
        assert len(self.observation_space.shape) == 2, "expect [n_agent x dim]"
        return self.observation_space.shape[1]

    def get_state(self):
        # copied from harl
        # adapted from imple115StateWrapper.convert_observation
        raw_state = self.env.unwrapped.observation()

        def do_flatten(obj):
            """Run flatten on either python list or numpy array."""
            if type(obj) == list:
                return np.array(obj).flatten()
            return obj.flatten()

        s = []
        for i, name in enumerate(
            ["left_team", "left_team_direction", "right_team", "right_team_direction"]
        ):
            s.extend(do_flatten(raw_state[0][name]))
            # If there were less than 11vs11 players we backfill missing values
            # with -1.
            if len(s) < (i + 1) * 22:
                s.extend([-1] * ((i + 1) * 22 - len(s)))
        # ball position
        s.extend(raw_state[0]["ball"])
        # ball direction
        s.extend(raw_state[0]["ball_direction"])
        # one hot encoding of which team owns the ball
        if raw_state[0]["ball_owned_team"] == -1:
            s.extend([1, 0, 0])
        if raw_state[0]["ball_owned_team"] == 0:
            s.extend([0, 1, 0])
        if raw_state[0]["ball_owned_team"] == 1:
            s.extend([0, 0, 1])
        game_mode = [0] * 7
        game_mode[raw_state[0]["game_mode"]] = 1
        s.extend(game_mode)
        for obs in raw_state:
            active = [0] * 11
            if obs["active"] != -1:
                active[obs["active"]] = 1
            s.extend(active)
        return np.array(s, dtype=np.float32)

    def get_state_size(self):
        """Returns the shape of the state"""
        # copied from harl
        # state space is designed following Simple115StateWrapper.convert_observation
        # global states are included once, and the active one-hot encodings for all players are included.
        total_length = 115 + (self.n_agents - 1) * 11
        return total_length

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take"""
        return np.max(self.action_space.nvec)

    def reset(self, seed=None, options=None):
        """Returns initial observations and info"""
        obs, info = self.env.reset()
        obs = self.split(obs)
        self._obs = obs
        self.previous_designated = None
        self._build_graph()
        
        return obs, info

    def _build_graph(self):
        self.graph = np.eye(self.n_agents)
        obs = self._env.unwrapped.observation()
        
        designated_player = obs[0]["designated"]
        if self.previous_designated is not None and designated_player != self.previous_designated:
            if self.previous_designated < self.n_agents and designated_player < self.n_agents:
                self.graph[self.previous_designated, designated_player] = 1
        self.previous_designated = designated_player
        return self.graph.copy()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.seed(seed)

    def split(self, a):
        return [a[i] for i in range(self.n_agents)]