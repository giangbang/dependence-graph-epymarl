from collections import defaultdict
import math

from lbforaging.foraging import ForagingEnv as BaseEnv
from lbforaging.foraging.environment import Action
import numpy as np
# from lbforaging.foraging.rendering import *


_WHITE = (255, 255, 255)

class ForagingEnv(BaseEnv):
    """LBF with winner-take-all rewards"""

    def step(self, actions):
        self.current_step += 1

        for p in self.players:
            p.reward = 0

        actions = [
            Action(a) if Action(a) in self._valid_actions[p] else Action.NONE
            for p, a in zip(self.players, actions)
        ]

        # check if actions are valid
        for i, (player, action) in enumerate(zip(self.players, actions)):
            if action not in self._valid_actions[player]:
                self.logger.info(
                    "{}{} attempted invalid action {}.".format(
                        player.name, player.position, action
                    )
                )
                actions[i] = Action.NONE

        loading_players = set()

        # move players
        # if two or more players try to move to the same location they all fail
        collisions = defaultdict(list)

        # so check for collisions
        for player, action in zip(self.players, actions):
            if action == Action.NONE:
                collisions[player.position].append(player)
            elif action == Action.NORTH:
                collisions[(player.position[0] - 1, player.position[1])].append(player)
            elif action == Action.SOUTH:
                collisions[(player.position[0] + 1, player.position[1])].append(player)
            elif action == Action.WEST:
                collisions[(player.position[0], player.position[1] - 1)].append(player)
            elif action == Action.EAST:
                collisions[(player.position[0], player.position[1] + 1)].append(player)
            elif action == Action.LOAD:
                collisions[player.position].append(player)
                loading_players.add(player)

        # and do movements for non colliding players
        for k, v in collisions.items():
            if len(v) > 1:  # make sure no more than an player will arrive at location
                continue
            v[0].position = k

        # finally process the loadings:
        while loading_players:
            # find adjacent food
            player = loading_players.pop()
            frow, fcol = self.adjacent_food_location(*player.position)
            food = self.field[frow, fcol]

            adj_players = self.adjacent_players(frow, fcol)
            adj_players = [
                p for p in adj_players if p in loading_players or p is player
            ]

            adj_player_level = sum([a.level for a in adj_players])
            loading_players = loading_players - set(adj_players)

            if adj_player_level < food:
                # failed to load
                for a in adj_players:
                    a.reward -= self.penalty
                continue

            # else the food was loaded and each player scores points
            for a in adj_players:
                a.reward = float(a.level * food)
                if self._normalize_reward:
                    a.reward = a.reward / float(
                        adj_player_level * self._food_spawned
                    )  # normalize reward
            # and the food is removed
            self.field[frow, fcol] = 0

            # winner-take-all rewards
            adj_levels = np.array([a.level for a in adj_players], dtype=int)
            adj_rws = np.array([a.reward for a in adj_players], dtype=np.float32)

            if len(adj_levels) > 0:
                # winners = np.argwhere(adj_levels == np.max(adj_levels))
                # winner = np.random.choice(winners.reshape(-1), size=1)[0]
                winner = np.argmax(adj_levels)
                for i, a in enumerate(adj_players):
                    if i == winner:
                        a.reward = np.sum(adj_rws)
                    else:
                        a.reward = 0
                assert np.sum(adj_rws) == np.sum([a.reward for a in adj_players])

        self._game_over = (
            self.field.sum() == 0 or self._max_episode_steps <= self.current_step
        )
        self._gen_valid_moves()

        for p in self.players:
            p.score += p.reward

        rewards = [p.reward for p in self.players]
        done = self._game_over
        truncated = False
        info = self._get_info()

        return self._make_gym_obs(), rewards, done, truncated, info
    
    def reset(self, seed=None, options=None):
        if seed is not None:
            # setting seed
            super().reset(seed=seed, options=options)

        if self.render_mode == "human":
            self.render()

        self.field = np.zeros(self.field_size, np.int32)
        self.spawn_players(self.min_player_level, self.max_player_level)
        player_levels = sorted([player.level for player in self.players])

        self.spawn_food(
            self.max_num_food,
            min_levels=self.min_food_level,
            max_levels=self.max_food_level
            if self.max_food_level is not None
            else np.array([sum(player_levels[:2])] * self.max_num_food),
        )
        self.current_step = 0
        self._game_over = False
        self._gen_valid_moves()

        nobs = self._make_gym_obs()
        return nobs, self._get_info()

    def render(self, graph=None, strength=None, **kwargs):
        import pyglet
        self.viewer.render(self, False)
        if graph is not None:
            self._draw_edges(self, graph, strength)

        buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        image_data = buffer.get_image_data()
        arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
        arr = arr.reshape(buffer.height, buffer.width, 4)
        arr = arr[::-1, :, 0:3]
        self.viewer.window.flip()
        return arr 


    def _draw_edges(self, env, graph, weights):
        """
        Draw arrows between agents based on `graph` adjacency matrix, with text labels from `weights`.
        """
        import pyglet
        n = len(env.players)
        coords = []
        # Get screen coordinates of each agent center
        for player in env.players:
            row, col = player.position
            x = (self.grid_size + 1) * col + self.grid_size / 2
            y = self.height - (self.grid_size + 1) * (row + 1) + self.grid_size / 2
            coords.append((x, y))

        for i in range(n):
            for j in range(n):
                if i == j or graph[i, j] == 0:
                    continue

                x1, y1 = coords[i]
                x2, y2 = coords[j]
                dx, dy = x2 - x1, y2 - y1
                dist = math.sqrt(dx**2 + dy**2)

                # Normalize to create arrow head
                if dist < 1e-5:
                    continue
                ux, uy = dx / dist, dy / dist

                # Shorten line a bit so arrows start/end at sprite edges
                margin = self.grid_size * 0.4
                start_x = x1 + ux * margin
                start_y = y1 + uy * margin
                end_x = x2 - ux * margin
                end_y = y2 - uy * margin

                # Draw the arrow shaft
                pyglet.graphics.draw(
                    2,
                    pyglet.gl.GL_LINES,
                    ('v2f', (start_x, start_y, end_x, end_y)),
                    ('c3B', (0, 0, 0, 0, 0, 0))  # black line
                )

                # Draw arrow head
                head_size = 6
                angle = math.atan2(dy, dx)
                left_x = end_x - head_size * math.cos(angle - math.pi / 6)
                left_y = end_y - head_size * math.sin(angle - math.pi / 6)
                right_x = end_x - head_size * math.cos(angle + math.pi / 6)
                right_y = end_y - head_size * math.sin(angle + math.pi / 6)

                pyglet.graphics.draw(
                    3,
                    pyglet.gl.GL_TRIANGLES,
                    ('v2f', (end_x, end_y, left_x, left_y, right_x, right_y)),
                    ('c3B', (0, 0, 0, 0, 0, 0, 0, 0, 0))
                )

                # Draw edge weight text (center of arrow)
                mid_x = (start_x + end_x) / 2
                mid_y = (start_y + end_y) / 2
                pyglet.text.Label(
                    f"{weights[i, j]:.2f}",
                    font_size=10,
                    x=mid_x,
                    y=mid_y,
                    anchor_x='center',
                    anchor_y='center',
                    color=(0, 0, 0, 255),
                ).draw()
    