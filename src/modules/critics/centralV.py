# code adapted from https://github.com/AnujMahajanOxf/MAVEN

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class CentralVCritic(nn.Module):
    def __init__(self, scheme, args):
        super(CentralVCritic, self).__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_shape = self._get_input_shape(scheme)
        self.output_type = "v"
        self.use_knn_graph = getattr(args, "use_knn_graph", False)
        # output_dim = self.n_agents if self.use_knn_graph else 1
        output_dim = 1
        if not self.use_cnn:

            # Set up network layers
            self.fc1 = nn.Linear(input_shape, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.fc3 = nn.Linear(args.hidden_dim, output_dim)
        else: 
            assert not self.use_cnn
            self.input_shape = input_shape
            from modules.critics.cnn import CNN
            self.net = CNN(input_shape, args.hidden_dim)
            self.fc1 = nn.Linear(args.hidden_dim+self.input_extra, args.hidden_dim)
            self.fc2 = nn.Linear(args.hidden_dim, output_dim)

    def forward(self, batch, t=None):
        if not self.use_cnn:
            inputs, bs, max_t = self._build_inputs(batch, t=t)
            x = F.relu(self.fc1(inputs))
            x = F.relu(self.fc2(x))
            q = self.fc3(x)
            return q
        else:
            cnn_x, mlp_x, bs, max_t = self._build_cnn_inputs(batch=batch, t=t)
            cnn_x = self.net(cnn_x)
            x = th.cat([cnn_x, mlp_x], dim=-1)

            x = F.relu(self.fc1(x))
            q = self.fc2(x)
            return q

    def _build_cnn_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)
        inputs = []
        # state

        # state
        inputs.append(
            batch["state"][:, ts]
            .unsqueeze(2)
            .repeat(1, 1, self.n_agents, *([1] * len(self.input_shape)))
        )

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs_mlp = th.cat(inputs[1:], dim=-1)
        return inputs[0], inputs_mlp, bs, max_t

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # state
        inputs.append(batch["state"][:, ts].unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # observations
        if self.args.obs_individual_obs:
            inputs.append(batch["obs"][:, ts].view(bs, max_t, -1).unsqueeze(2).repeat(1, 1, self.n_agents, 1))

        # last actions
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, 0:1]).view(bs, max_t, 1, -1))
            elif isinstance(t, int):
                inputs.append(batch["actions_onehot"][:, slice(t-1, t)].view(bs, max_t, 1, -1))
            else:
                last_actions = th.cat([th.zeros_like(batch["actions_onehot"][:, 0:1]), batch["actions_onehot"][:, :-1]], dim=1)
                last_actions = last_actions.view(bs, max_t, 1, -1).repeat(1, 1, self.n_agents, 1)
                inputs.append(last_actions)

        inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

        inputs = th.cat(inputs, dim=-1)
        return inputs, bs, max_t

    def _get_input_shape(self, scheme):
        # state
        input_shape = scheme["state"]["vshape"]
        self.use_cnn = not isinstance(input_shape, int)
        input_extra = 0
        # print("input_shape", input_shape)
        # observations
        if self.args.obs_individual_obs:
            input_extra += scheme["obs"]["vshape"] * self.n_agents
        # last actions
        if self.args.obs_last_action:
            input_extra += scheme["actions_onehot"]["vshape"][0] * self.n_agents
        input_extra += self.n_agents
        self.input_extra = input_extra
        if self.use_cnn:
            return input_shape
        else:
            return input_shape + input_extra
