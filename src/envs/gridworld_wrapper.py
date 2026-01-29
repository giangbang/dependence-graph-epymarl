import numpy as np
from envs.gridworld.gridworld_env import GridworldEnv
from envs.multiagentenv import MultiAgentEnv


class GridworldWrapper(MultiAgentEnv):
    def __init__(
        self,
        seed,
        key,
        common_reward,
        reward_scalarisation="sum",
        **kwargs,
    ):
        # key of the form gridworld:plan_1
        self.common_reward = common_reward
        if "plan" in key:
            plan = int(key.split("_")[-1])
            self.env: GridworldEnv = GridworldEnv(plan=plan, seed=seed, separated_rewards=not common_reward)
        else:
            from envs.gridworld.n_room.n_room import NRoom
            nroom = int(key.split("_")[-1]) 
            self.env: NRoom = NRoom(nroom=nroom, seed=seed, separated_rewards=not common_reward)
        self.n_agents = self.env.n_agents
        self.episode_limit = self.env.max_step
        self.reset()

    def reset(self):
        obs, states, _ = self.env.reset()

        self.obs = obs
        self.states = states
        return obs, {}

    def step(self, actions):
        obs, states, rewards, done, infos, _ = self.env.step(actions)
        self.obs = obs
        self.states = states
        
        if self.common_reward:
            assert isinstance(rewards, (int, float)), rewards
        else:
            assert np.prod(rewards.shape) == self.n_agents, rewards.shape
            rewards = rewards.squeeze()

        return obs, rewards, False, done, {}

    def get_obs(self):
        return self.obs
    
    def get_obs_agent(self, agent_id):
        return self.obs[agent_id]
    
    def get_obs_size(self):
        shape = self.obs[0].shape
        assert len(shape) == 1
        return shape[0]

    def get_state(self):
        return np.concatenate(self.obs, axis=0).astype(np.float32)

    def get_state_size(self):
        # shape = self.states.shape
        # assert len(shape) == 1
        # return shape[0]
        return self.obs[0].shape[0] * self.n_agents
    
    def get_avail_actions(self):
        return np.ones((self.n_agents, self.get_total_actions()), dtype=bool)
    
    def close(self):
        return

    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.get_total_actions(), dtype=bool)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)
    
    def get_total_actions(self):
        return len(self.env.actions)
    
    def seed(self, seed):
        return self.env.seed(seed)
    