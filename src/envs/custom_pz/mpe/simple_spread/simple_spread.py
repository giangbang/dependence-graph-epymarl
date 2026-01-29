import numpy as np
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn


class raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        N=3,
        local_ratio=0.5,
        max_cycles=25,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
    ):
        EzPickle.__init__(
            self,
            N=N,
            local_ratio=local_ratio,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        assert (
            0.0 <= local_ratio <= 1.0
        ), "local_ratio is a proportion. Must be between 0 and 1."
        scenario = Scenario(local_ratio=local_ratio)
        world = scenario.make_world(N)
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            local_ratio=local_ratio,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.metadata["name"] = "simple_spread_v3"


env = make_env(raw_env)
parallel_env = parallel_wrapper_fn(env)

from pettingzoo.mpe.simple_spread.simple_spread import Scenario as SpreadScenario


class Scenario(SpreadScenario):
    def __init__(self, local_ratio, **kwargs):
        super().__init__(**kwargs)
        self.local_ratio = local_ratio

    def reward(self, agent: Agent, world: World):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        agent_id = world.agents.index(agent)
        if agent.collide:
            for a in world.agents:
                rew -= 1.0 * (self.is_collision(a, agent) and a != agent)

        # global reward equivalent, denpend only on the closest agents' distances to landmarks
        global_rew = 0
        for lm in world.landmarks:
            dists = [
                np.sum(np.square(a.state.p_pos - lm.state.p_pos)) for a in world.agents
            ]
            dists = np.sqrt(dists)

            nearest_agent = dists.argmin()
            nearest_dist = np.min(dists)

            if nearest_agent == agent_id:
                global_rew -= nearest_dist

        global_rew *= len(world.agents)  # total rewards are distribution to this agent only
        final_rew = self.local_ratio * rew + (1 - self.local_ratio) * global_rew

        return final_rew / self.local_ratio  # revert `local_ratio` later

    def global_reward(self, world):
        return 0
