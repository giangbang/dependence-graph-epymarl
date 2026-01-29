import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, scheme, args):
        super().__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        self.one_hot_agent = False

        input_shape = self._get_input_shape(scheme)
        output_dim = args.latten_state_dim
        self.single_agent_reverse_model = SingleAgentReverseModel(output_dim, self.n_actions, scheme, args)
        if not self.use_cnn:

            # Set up network layers
            self.net = nn.Sequential(
                nn.Linear(input_shape, args.state_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(args.state_hidden_dim),
                nn.Linear(args.state_hidden_dim, args.state_hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(args.state_hidden_dim),
                nn.Linear(args.state_hidden_dim, output_dim)
            )
        else: 
            assert False, "not supported"

    def forward(self, obs):
        return self.net(obs)

    def _build_inputs(self, batch, key="obs", t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t+1)
        inputs = []
        # observations
        inputs.append(batch[key][:, ts])

        if not self.use_cnn:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).unsqueeze(0).expand(bs, max_t, -1, -1))

            inputs = th.cat(inputs, dim=-1)
            return inputs, bs, max_t
        else:
            return inputs[0], bs, max_t

    def _get_input_shape(self, scheme):
        # observations
        input_shape = scheme["obs"]["vshape"]
        self.use_cnn = not isinstance(input_shape, int)
        # agent id
        # if not self.use_cnn:
        #     input_shape += self.n_agents
        return input_shape


class MultiAgentReverseModel(nn.Module):
    """
    Multi-agent Reverse world model
    take input the tuple (s^i, s^i', s^j), predict the action a^j
    """
    def __init__(self, input_dim, n_actions, scheme, args, n_ensemble=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, args.revs_model_hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim * 2),
            nn.Linear(args.revs_model_hidden_dim * 2, args.revs_model_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, args.revs_model_hidden_dim),
            nn.ReLU(), 
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, n_actions)
        )


        self.n_actions = n_actions
        self.n_agents = args.n_agents

    def forward(self, current_state, next_state):
        """return logits matrix A of shape nxn
        where A[i, j] denote tuple (s^j, s^j', s^i)"""
        b, t, n, _ = current_state.shape
        inputs = th.cat(
                (current_state.repeat((1, 1, self.n_agents, 1)),
                 next_state.repeat((1, 1, self.n_agents, 1)),
                 current_state.repeat_interleave(self.n_agents, dim=-2)
                ), 
            dim=-1
        )
        return self.net(inputs).view(b, t, n, n, self.n_actions)

    def revrs_loss(self, state, action):
        # batch, time, agent, dim
        st = state
        curr_state = st[:, :-1]
        next_state = st[:, 1:]
        action = action[:, :-1].repeat_interleave(self.n_agents, dim=-2)

        pred = self(curr_state, next_state)
        loss = F.cross_entropy(pred.reshape(-1, self.n_actions), action.reshape(-1))

        return loss

class SingleAgentReverseModel(nn.Module):
    """
    Single-agent Reverse world model
    take input the tuple (s^i, s^i'), predict the action a^i
    """
    def __init__(self, input_dim, n_actions, scheme, args):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, args.revs_model_hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim * 2),
            nn.Linear(args.revs_model_hidden_dim * 2, args.revs_model_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, args.revs_model_hidden_dim),
            nn.ReLU(), 
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, n_actions)
        )
        self.n_actions = n_actions

    def forward(self, current_state, next_state):
        inputs = th.cat((current_state, next_state), dim=-1)
        return self.net(inputs)

    def revrs_loss(self, state, action):
        # batch, time, agent, dim
        st = state
        curr_state = st[:, :-1]
        next_state = st[:, 1:]
        action = action[:, :-1]

        pred = self(curr_state, next_state)
        loss = F.cross_entropy(pred.reshape(-1, self.n_actions), action.reshape(-1))

        return loss

class ActionPredictor(nn.Module):
    """
    Given s_i, predict a_i
    """
    def __init__(self, input_dim, n_actions, scheme, args):
        super().__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, args.revs_model_hidden_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim * 2),
            nn.Linear(args.revs_model_hidden_dim * 2, args.revs_model_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, args.revs_model_hidden_dim),
            nn.ReLU(), 
            nn.LayerNorm(args.revs_model_hidden_dim),
            nn.Linear(args.revs_model_hidden_dim, n_actions)
        )

    def forward(self, state):
        return self.net(state)

    def pred_loss(self, state, action):
        action_pred = self(state).view(-1, self.n_actions)
        loss = F.cross_entropy(action_pred, action.view(-1))
        return loss

class GraphModel(nn.Module):

    def __init__(self, scheme, args):
        super().__init__()

        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents

        input_dim = args.latten_state_dim
        self.multi_agent_reverse_model = MultiAgentReverseModel(input_dim, self.n_actions, scheme, args)
        self.action_predictor = ActionPredictor(input_dim, self.n_actions, scheme, args)

    def fullbatch_forward(self, obs, state_encoder, device):
        obs = obs.to(device)
        # onehot = th.eye(self.n_agents, device=device).view(1, 1, self.n_agents, self.n_agents)
        # onehot = onehot.expand(thread, eps_len, -1, -1)

        # obs = th.cat((obs, onehot), dim=-1)

        with th.no_grad():
            state = state_encoder(obs).detach()
            current_state = state[:, :-1]
            next_state = state[:, 1:]

            marginal_logits = self.action_predictor(current_state)
            marginal_entropy = th.distributions.Categorical(logits=marginal_logits).entropy()

            conditional_logits = self.multi_agent_reverse_model(current_state, next_state)
            conditional_entropy = th.distributions.Categorical(logits=conditional_logits).entropy()

        return marginal_entropy, conditional_entropy.permute(0, 1, 3, 2)
