import gym
import numpy as np


class LocalRewardFootballWrapper(gym.RewardWrapper):

    def reward(self, reward):
        # observation = self.env.unwrapped.observation()
        # rew = reward[0].item()
        # assert np.abs(rew - np.mean(reward)) < 1e-6, (
        #     f"{reward} is expected to "
        #     "be a contant vector, recheck env's reward sepcification."
        # )
        # returned_rew = np.zeros_like(reward)
        # designated_player = [o["designated"] for o in observation]
        # assert np.all((d == designated_player[0] for d in designated_player))
        # returned_rew[designated_player] = rew
        # return returned_rew
        return reward