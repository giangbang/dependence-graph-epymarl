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
        self.use_encoder = args.use_encoder

        self.one_hot_agent = False

        input_shape = self._get_input_shape(scheme)
        if not self.use_encoder:
            args.latten_state_dim = input_shape
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
        if self.use_encoder:
            return self.net(obs)
        else: return obs

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
    multi_head: each agent has its own head to predict its own action, otherwise one shared head
    """
    def __init__(self, input_dim, n_actions, scheme, args, 
                  n_ensemble=1):
        super().__init__()
        multi_head = args.rev_model_multi_head
        self.output_dim = n_actions if not multi_head else n_actions * args.n_agents
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
            nn.Linear(args.revs_model_hidden_dim, self.output_dim)
        )
        self.multi_head = multi_head


        self.n_actions = n_actions
        self.n_agents = args.n_agents

    def forward(self, current_state, next_state):
        """return logits matrix A of shape nxn
        where A[i, j] denote tuple (s^j, s^j', s^i)"""
        if len(current_state.shape) == 4:
            b, t, n, _ = current_state.shape
            assert n == self.n_agents, f"{n} {self.n_agents}"
            inputs = th.cat(
                    (current_state.repeat((1, 1, self.n_agents, 1)),
                    next_state.repeat((1, 1, self.n_agents, 1)),
                    current_state.repeat_interleave(self.n_agents, dim=-2)
                    ), 
                dim=-1
            )
            out = self.net(inputs)
            if not self.multi_head:
                out = out.view(b, t, n, n, self.n_actions)
                return out
            else:
                out = out.view(b, t, n, n, self.n_agents, self.n_actions)
                # out shape: [10, 36, 6, 6, 6, 30] -> dims (a, b, i, j, i2, k)
                # torch.diagonal takes the diagonal between dim2 and dim4, appends it as last dim
                out = out.diagonal(dim1=2, dim2=4)  # shape: [10, 36, 6, 30, 6] -> (a, b, j, k, i)
                out = out.permute(0, 1, 4, 2, 3)  # -> (a, b, i, j, k) = [10, 36, 6, 6, 30]
                return out
        elif len(current_state.shape) == 3:
            b, n, _ = current_state.shape
            assert n == self.n_agents, f"{n} {self.n_agents} {current_state.shape}"
            inputs = th.cat(
                    (current_state.repeat((1, self.n_agents, 1)),
                    next_state.repeat((1, self.n_agents, 1)),
                    current_state.repeat_interleave(self.n_agents, dim=-2)
                    ), 
                dim=-1
            )
            out = self.net(inputs)
            if not self.multi_head:
                out = out.view(b, n, n, self.n_actions)
                return out
            else:
                out = out.view(b, n, n, self.n_agents, self.n_actions)
                # (b, i, j, i2, k) -> diagonal over i,i2 -> (b, j, k, i) -> permute -> (b, i, j, k)
                out = out.diagonal(dim1=-4, dim2=-2).permute(0, 3, 1, 2)
                return out
        else:
            raise Exception()

    def revrs_loss(self, state, action, mask):
        # batch, time, agent, dim
        st = state
        curr_state = st[:, :-1]
        next_state = st[:, 1:]
        action = action[:, :-1].repeat_interleave(self.n_agents, dim=-2)
        if mask is not None:
            indx = mask[:, :-1].bool().squeeze(-1).nonzero(as_tuple=True)

            curr_state = curr_state[indx]
            action = action[indx]
            next_state = next_state[indx]


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

    def revrs_loss(self, state, action, mask=None):

        # batch, time, agent, dim
        st = state
        curr_state = st[:, :-1]
        next_state = st[:, 1:]
        action = action[:, :-1]

        if mask is not None:
            indx = mask[:, :-1].bool().squeeze(-1).nonzero(as_tuple=True)
            
            curr_state = curr_state[indx]
            action = action[indx]
            next_state = next_state[indx]


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

    def pred_loss(self, state, action, mask):
        if mask is not None:
            indx = mask.bool().squeeze(-1).nonzero(as_tuple=True)
            
            state = state[indx]
            action = action[indx]

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
