from collections.abc import Iterable

import warnings
from .stag_hunt import StagHunt


class StagHuntWrapper(StagHunt):
    def __init__(self, common_reward, reward_scalarisation, **kwargs):
        super().__init__(**kwargs)
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
        # actions = [int(a) for a in actions]
        obs, reward, done, truncated, self._info = super().step(actions)
        self._obs = obs

        if self.common_reward and isinstance(reward, Iterable):
            reward = float(self.reward_agg_fn(reward))
        elif not self.common_reward and not isinstance(reward, Iterable):
            warnings.warn(
                "common_reward is False but received scalar reward from the environment, returning reward as is"
            )

        if isinstance(done, Iterable):
            done = all(done)
        return self._obs, reward, done, truncated, self._info