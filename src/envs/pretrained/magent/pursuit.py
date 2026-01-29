from functools import partial
import math
from enum import IntEnum
import random
import cv2
from gymnasium.spaces import Tuple
import gymnasium as gym
import numpy as np
import pygame


class Pursuit(gym.Wrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from magent2.environments.magent_env import magent_parallel_env
        from envs.magent_wrapper import MAgentEnv

        env: magent_parallel_env = self.env.unwrapped._env.unwrapped
        self.magent_paralle_env = env
        self.gym_env: MAgentEnv = self.env.unwrapped

        self.tot_agents = len(env.possible_agents)
        self.n_agents = 0  # persuit agents, to be trained with RL
        for ac in self.action_space:
            if ac.n > 10:
                self.n_agents += 1
        # bot agents
        self.n_pt_agents = self.tot_agents - self.n_agents

        self.pt_action_space = self.action_space[-self.n_pt_agents:]
        self.pt_observation_space = self.observation_space[-self.n_pt_agents:]

        self.action_space = Tuple(self.action_space[:self.n_agents])
        self.observation_space = Tuple(self.observation_space[:self.n_agents])

        self.bot_agents = SimpleVectorizedPrey()
        self.prev_obs = None
        self.prev_mask = None

        self.flatten_obs = True
        if self.flatten_obs:
            self._original_obs_shape = [obs_space.shape for obs_space in self.observation_space]
            self.observation_space = Tuple(
                tuple([
                    gym.spaces.Box(low=-1, high=1, 
                                shape=(2 + np.prod(obs_shape).item(),)) 
                    for obs_shape in self._original_obs_shape
                ])
            )

    def reset(self, seed=None, options=None):
        obss, info = super().reset(seed=seed, options=options)
        self.prev_obs = obss[self.n_agents :]
        ret_obss = obss[:self.n_agents]
        if self.flatten_obs:
            ret_obss = self._append_pos_to_obs(ret_obss)
        self.prev_mask = np.ones(self.n_agents, dtype=bool)
        return ret_obss, info

    def _append_pos_to_obs(self, obss):
        if not self.flatten_obs:
            return obss
        all_pos = self.gym_env.get_pos()
        pos_vec = self.gym_env.construct_pos_vec(all_pos)
        agent_pos = pos_vec[:self.n_agents]
        obss = np.array(obss)
        obss = obss.reshape(self.n_agents, -1)
        cat_obss = np.concatenate((obss, agent_pos), axis=-1)
        return cat_obss

    def step(self, action):
        self.prev_obs = np.array(self.prev_obs)
        bot_action = tuple(self.bot_agents.act(self.prev_obs))
        action = tuple(action) + bot_action
        obs, rew, done, truncated, info = super().step(action)

        self.prev_obs = obs[self.n_agents :]
        self.prev_mask = done or truncated
        obs = obs[:self.n_agents]
        rew = rew[:self.n_agents]

        if self.flatten_obs:
            obs = self._append_pos_to_obs(obs)

        return obs, rew, done, truncated, info

    def render(self, graph=None, strength=None):
        if graph is not None:
            # print(graph)
            graph = np.concatenate((graph, np.zeros((self.n_agents, self.n_pt_agents))), axis=-1)
            graph = np.concatenate((graph, np.zeros((self.n_pt_agents, self.tot_agents))), axis=-2)

            draw_fn = partial(draw_graph_fn, graph=graph)
            rgb = self.magent_paralle_env.render(draw_graph_fn=draw_fn)
        else:
            return super().render()

        return rgb


def draw_arrow_head(surface, start, end, color=(0, 255, 0), arrow_size=8):
    """Draws a small arrowhead pointing from start → end"""
    dx, dy = end[0] - start[0], end[1] - start[1]
    angle = math.atan2(dy, dx)
    # Compute two lines forming the arrow tip
    left = (end[0] - arrow_size * math.cos(angle - math.pi / 6),
            end[1] - arrow_size * math.sin(angle - math.pi / 6))
    right = (end[0] - arrow_size * math.cos(angle + math.pi / 6),
             end[1] - arrow_size * math.sin(angle + math.pi / 6))
    pygame.draw.polygon(surface, color, [end, left, right])


def draw_graph_fn(renderer, graph):
    grid_size = 8

    self = renderer

    # Collect agent positions for later
    agent_positions = {}

    resolution = self.resolution

    view_position = [
            self.map_size[0] / 2 * grid_size - resolution[0] / 2,
            self.map_size[1] / 2 * grid_size - resolution[1] / 2,
        ]

    for agent_id, agent_data in self.new_data[0].items():
        x, y, group_id = agent_data[0], agent_data[1], agent_data[2]
        agent_positions[agent_id] = (
            x * grid_size - view_position[0] + grid_size / 2,
            y * grid_size - view_position[1] + grid_size / 2,
        )

    # Suppose self.graph is your n_agent x n_agent matrix
    # or replace with your own adjacency matrix variable

    n_agent = len(graph)
    arrow_color = (0, 200, 0)  # green arrows

    for i in range(n_agent):
        for j in range(n_agent):
            if i == j: continue
            if graph[i, j] > 0 and i in agent_positions and j in agent_positions:
                start = agent_positions[i]
                end = agent_positions[j]

                # Draw main line
                pygame.draw.line(self.canvas, arrow_color, start, end, 2)

                # Draw arrowhead
                draw_arrow_head(self.canvas, start, end, color=arrow_color)


class PreyAction(IntEnum):
    MOVE_UP_LEFT = 0
    MOVE_UP = 1
    MOVE_UP_RIGHT = 2
    MOVE_LEFT = 3
    DO_NOTHING = 4
    MOVE_RIGHT = 5
    MOVE_DOWN_LEFT = 6
    MOVE_DOWN = 7
    MOVE_DOWN_RIGHT = 8
    
class PredatorAction(IntEnum):
    MOVE_UP = 0
    MOVE_LEFT = 1
    DO_NOTHING = 2
    MOVE_RIGHT = 3
    MOVE_DOWN = 4

    # predator occupy 4 squares
    # attack up left is attacking the upper row, the left cell
    TAG_UP_LEFT = 5
    TAG_UP_RIGHT = 6
    TAG_LEFT_UP = 7
    TAG_RIGHT_UP = 8
    TAG_LEFT_DOWN = 9
    TAG_RIGHT_DOWN = 10
    TAG_DOWN_LEFT = 11
    TAG_DOWN_RIGHT = 12

class SimplePrey:
    def __init__(self, other_channel=3, flee_prob=0.6):
        self.other = other_channel
        self.flee_prob = flee_prob

    def act(self, obs):
        H, W, C = obs.shape
        r, c = H // 2 - 1, W // 2 - 1
        layer = obs[..., self.other]
        ys, xs = np.nonzero(layer)
        if ys.size == 0 or random.random() > self.flee_prob:
            return int(random.choice(list(PreyAction)))
        dists = np.abs(ys - r) + np.abs(xs - c)
        i = int(dists.argmin())
        pr, pc = int(ys[i]), int(xs[i])
        dr, dc = r - pr, c - pc
        sdr = 0 if dr == 0 else (1 if dr > 0 else -1)
        sdc = 0 if dc == 0 else (1 if dc > 0 else -1)
        for act, (dy, dx) in {
            PreyAction.MOVE_UP_LEFT:(-1,-1),
            PreyAction.MOVE_UP:(-1,0),
            PreyAction.MOVE_UP_RIGHT:(-1,1),
            PreyAction.MOVE_LEFT:(0,-1),
            PreyAction.DO_NOTHING:(0,0),
            PreyAction.MOVE_RIGHT:(0,1),
            PreyAction.MOVE_DOWN_LEFT:(1,-1),
            PreyAction.MOVE_DOWN:(1,0),
            PreyAction.MOVE_DOWN_RIGHT:(1,1),
        }.items():
            if (dy, dx) == (sdr, sdc):
                return int(act)
        return int(PreyAction.DO_NOTHING)


class SimplePredator:
    def __init__(self, other_channel=3):
        self.other = other_channel

    def act(self, obs):
        H, W, C = obs.shape
        r, c = H // 2, W // 2
        layer = obs[..., self.other]
        ys, xs = np.nonzero(layer)
        if ys.size == 0:
            return int(random.choice([
                PredatorAction.MOVE_UP,
                PredatorAction.MOVE_LEFT,
                PredatorAction.DO_NOTHING,
                PredatorAction.MOVE_RIGHT,
                PredatorAction.MOVE_DOWN,
            ]))
        dists = np.abs(ys - r) + np.abs(xs - c)
        i = int(dists.argmin())
        pr, pc = int(ys[i]), int(xs[i])
        if pr == r-2 and pc == c:     return int(PredatorAction.TAG_UP_RIGHT)
        if pr == r-2 and pc == c-1:   return int(PredatorAction.TAG_UP_LEFT)
        if pr == r-1 and pc == c-2:   return int(PredatorAction.TAG_LEFT_UP)
        if pr == r   and pc == c-2:   return int(PredatorAction.TAG_LEFT_DOWN)
        if pr == r-1 and pc == c+1:   return int(PredatorAction.TAG_RIGHT_UP)
        if pr == r   and pc == c+1:   return int(PredatorAction.TAG_RIGHT_DOWN)
        if pr == r+1 and pc == c-1:   return int(PredatorAction.TAG_DOWN_LEFT)
        if pr == r+1 and pc == c:     return int(PredatorAction.TAG_DOWN_RIGHT)
        dr, dc = pr - r, pc - c
        if abs(dc) > abs(dr):
            return int(PredatorAction.MOVE_RIGHT if dc > 0 else PredatorAction.MOVE_LEFT)
        return int(PredatorAction.MOVE_DOWN if dr > 0 else PredatorAction.MOVE_UP)



class SimpleVectorizedPrey:
    def __init__(self, other_channel=3, flee_prob=0.6, rng=None):
        self.other = other_channel
        self.flee_prob = flee_prob
        self.rng = rng if rng is not None else np.random.default_rng()
        self.map_array = np.array([
            [0,1,2],
            [3,4,5],
            [6,7,8]
        ])
        self.eps_random = 0.1

    def act(self, obs):
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        single = False
        if obs.ndim == 3:
            obs = obs[None,...]
            single = True
        N,H,W,_ = obs.shape
        r,c = H//2-1, W//2-1
        mask = obs[...,self.other]!=0
        any_pred = mask.reshape(N,-1).any(1)
        flee = (self.rng.random(N) <= self.flee_prob) & any_pred
        actions = self.rng.integers(0,9,N)
        if flee.any():
            ys,xs = np.arange(H)[:,None], np.arange(W)[None,:]
            dist = np.abs(ys-r)+np.abs(xs-c)
            idxs = np.where(flee)[0]
            flat = np.where(mask[idxs],dist,max(H,W)+10).reshape(len(idxs),-1)
            arg = flat.argmin(1)
            pr,pc = arg//W,arg%W
            sdr,sdc = np.sign(r-pr),np.sign(c-pc)
            actions[idxs] = self.map_array[sdr+1,sdc+1]
        # actions = self.pertube_action(actions)
        return int(actions[0]) if single else actions

    def pertube_action(self, action):
        n = len(action)
        num_to_replace = int(n * self.eps_random / 100)
        indices = self.rng.choice(n, num_to_replace, replace=False)
        action[indices] = np.random.randint(0, 10, size=num_to_replace)
        return action

