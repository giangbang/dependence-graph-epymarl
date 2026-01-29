import os
import numpy as np

from learners.ppo_graph_learner import PPOGraphLearner
from .episode_runner import EpisodeRunner
from envs import REGISTRY as env_REGISTRY
import cv2
import torch
import copy


class VisualizeRunner(EpisodeRunner):
    def __init__(self, args, logger):
        args = copy.deepcopy(args)
        args.batch_size = 1
        args.batch_size_run = 1
        super().__init__(args, logger)
        try:
            render_env = env_REGISTRY[self.args.env](
                **self.args.env_args,
                common_reward=self.args.common_reward,
                reward_scalarisation=self.args.reward_scalarisation,
                render_mode="rgb_array",
            )
        except:
            render_env = self.env
        self.env = render_env

    def _render(self):
        try:
            return self.env.render()
        except:
            return self.env.render(render_mode="rgb_array")

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, learner = None):
        self.reset()
        rendered_imgs = [self._render()]

        terminated = False
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:
            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            with torch.no_grad():
                actions, mac_out = self.mac.select_actions(
                    self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, return_mac_output=True
                )
            cpu_actions = actions.to("cpu").numpy()
            # mac_out = mac_out.to("cpu").numpy()

            _, reward, terminated, truncated, env_info = self.env.step(cpu_actions[0])
            terminated = terminated or truncated
            if test_mode and self.args.render:
                self.env.render()

            post_transition_data = {
                "actions": actions,
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }
            if self.args.common_reward:
                post_transition_data["reward"] = [(reward,)]
            else:
                post_transition_data["reward"] = [tuple(reward)]

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1
            
            if learner is not None and isinstance(learner, PPOGraphLearner):
                if self.t == 1: continue  # need s and s' for graph
                with torch.no_grad():
                    graph, ratio, _, _ = learner.infer_graph(self.batch, t_ep=self.t-1)

                rendered_imgs.append(self.env.render(graph=graph.squeeze().to("cpu").numpy(), 
                                                     strength=ratio.squeeze().to("cpu").numpy()))
            else:
                rendered_imgs.append(self._render())
        
        return rendered_imgs

