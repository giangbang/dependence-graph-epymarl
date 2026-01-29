import numpy as np
import torch as th


def advantage_graph(advantages, graph, gamma):
    # advantage: batch x timestep x n_agent
    # rewards: n_batch x timestep x n_agent
    # values : n_batch x timestep x n_agent
    # graph  : n_batch x timesetp x n_agent x n_agent
    next_advantages = th.zeros_like(advantages)

    next_advantages[:, :-1] = advantages[:, 1:]
    delta = advantages - gamma * next_advantages
    timestep = advantages.size(1)

    ret = th.zeros_like(graph)
    for t in range(timestep - 2, -1, -1):  # t = T-1, T-2. ..., 0
        ret[:, t] = gamma * ret[:, t + 1] + (delta[:, t].unsqueeze(-2) * graph[:, t])
    return ret.sum(dim=-1)


# def compute_adv_gae_graph_fullpath(
#     gamma, rewards, mask, vals, gae_lambda, graph, skip_ts
# ):
#     """graph [i, j] = 1: agent i has impact to agent j"""
#     skip_ts=1  # legacy, not used
#     # vals : n_batch x timestep x n_agent
#     timestep = int(vals.size(1)) - 1
#     advantages = th.zeros_like(rewards)

#     # graph  : n_batch x (timesetp-1) x n_agent x n_agent
#     # graph[i, j]: agent i -> j
#     or_graph = graph.detach().clone()
#     paths = th.zeros_like(graph)

#     paths[:, -1] = graph[:, -1]
#     # current_path_shape = paths[:, t + 1].shape
#     for t in reversed(range(timestep)):
#         graph[:, t] = or_graph[:, (t // skip_ts) * skip_ts]
#     for t in reversed(range(timestep - 1)):
#         current_path = graph[:, t].clone()
#         for t_prime in range(t + 1, timestep):
#             current_path = th.max(current_path, graph[:, t_prime])
#         paths[:, t] = current_path

#     gae = 0
#     for t in reversed(range(timestep)):
#         delta = (
#             rewards[:, t] * mask[:, t]
#             + gamma * vals[:, t + 1] * mask[:, t + 1]
#             - vals[:, t]
#         )
        
#         # path: n_batch x timestep x n_agent x n_agent
#         # delta: n_batch x n_agent
        
#         # delta = ((rewards[:, t] * mask[:, t]).unsqueeze(-2) * graph[:, t]).sum(-1) + gamma * vals[:, t + 1] * mask[:, t + 1] - vals[:, t]
#         gae = (delta.unsqueeze(-2) * (paths[:, t] * gae_lambda + 1 - gae_lambda)).sum(
#             -1
#         ) + gamma * gae_lambda * mask[:, t + 1] * gae
#         # gae = delta + gamma * gae_lambda * mask[:, t + 1] * gae
#         advantages[:, t] = gae.clone()

#     return advantages


def compute_adv_gae_graph_naive(
        gamma, rewards, mask, vals, gae_lambda, graph, skip_ts
):
    # vals : n_batch x timestep x n_agent
    timestep = int(vals.size(1)) - 1
    advantages = th.zeros_like(rewards)
    
    # cum_mask = th.zeros_like(mask)
    # cum_mask[:, -1] = mask[:, -1]

    # for t in reversed(range(1, timestep)):
    #     cum_mask[:, t - 1] = th.max(cum_mask[:, t], mask[:, t-1])


    delta_t = []
    for t in range(timestep):
        delta = (
                rewards[:, t] * mask[:, t]
                + gamma * vals[:, t + 1] * mask[:, t + 1]
                - vals[:, t]
            )
        delta_t.append(delta)
    
    graph = graph.float()

    for t_0 in reversed(range(timestep)):
        gae = 0
        # reach = th.zeros_like(graph[:, t_0])
        # reach = th.ones_like(graph[:, t_0])
        # reach.diagonal(dim1=-2, dim2=-1).fill_(1)
        reach = graph[:, t_0].clone()
        power = reach.clone()
        # current_mask = th.ones_like(mask[:, t_0])
        current_mask = mask[:, t_0].clone()

        for t in range(t_0, min(50 + t_0, timestep)):
            delta = delta_t[t]
            # assert delta.unsqueeze(-2).shape == reach.shape, f"{delta.shape} {reach.shape}"

            gae_now = (delta.unsqueeze(-2) * (reach* gae_lambda + 1 - gae_lambda)).sum(
                -1
            ) * ((gamma * gae_lambda) ** (t - t_0))
            assert gae_now.shape == current_mask.shape, f"{gae_now.shape} {current_mask.shape}"

            gae = gae_now * current_mask + gae
            current_mask = th.min(current_mask, mask[:, t + 1])
            # gae = delta + gamma * gae_lambda * mask[:, t + 1] * gae
            # power = th.bmm(power, graph[:, min(t+1, timestep-1)])
            power = th.bmm(power, graph[:, t])
            # power = th.maximum(power, graph[:, t])
            power = th.clamp(power, min=0, max=1)
            reach = th.clamp(reach + power, min=0, max=1)
        advantages[:, t_0] = gae

    return advantages


# def compute_adv_gae_graph(gamma, rewards, mask, vals, gae_lambda, graph, skip_ts=1):
#     # vals : n_batch x timestep x n_agent x n_agent
#     timestep = int(vals.size(1)) - 1
#     advantages = th.zeros_like(rewards)

#     gae = 0
#     for t in reversed(range(timestep)):
#         delta = (
#             rewards[:, t] * mask[:, t]
#             + gamma * vals[:, t + 1] * mask[:, t + 1]
#             - vals[:, t]
#         )
#         # delta = ((rewards[:, t] * mask[:, t]).unsqueeze(-2) * graph[:, t]).sum(-1) + gamma * vals[:, t + 1] * mask[:, t + 1] - vals[:, t]
#         gae = (delta.unsqueeze(-2) * \
#                 (graph[:, (t // skip_ts) * skip_ts] * gae_lambda + 1 - gae_lambda)
#             ).sum(
#             -1
#         ) + gamma * gae_lambda * mask[:, t + 1] * gae
#         # gae = delta + gamma * gae_lambda * mask[:, t + 1] * gae
#         advantages[:, t] = gae.clone()

#     return advantages


def compute_adv_gae(gamma, rewards, mask, vals, gae_lambda):
    timestep = int(vals.size(1)) - 1
    advantages = th.zeros_like(rewards)

    gae = 0
    for t in reversed(range(timestep)):

        delta = rewards[:, t] * mask[:, t] + gamma * vals[:, t + 1] * mask[:, t + 1] - vals[:, t]
        gae = delta + gamma * gae_lambda * mask[:, t + 1] * gae
        # advantages[:, t] = (gae.unsqueeze(-2)
        advantages[:, t] = gae.clone()

    return advantages


def compute_adv_nstep(gamma, rewards, mask, target_vals, vals):
    timestep = int(target_vals.size(1)) - 1
    advantages = th.zeros_like(target_vals[:, :-1])
    returns = vals[:, -1]

    for t in reversed(range(timestep)):
        returns = rewards[:, t] + gamma * returns * mask[:, t + 1]
        advantages[:, t] = returns.clone()
    return advantages
