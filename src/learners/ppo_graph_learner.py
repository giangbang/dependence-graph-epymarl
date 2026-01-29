import numpy as np
from learners import utils
from components.episode_buffer import EpisodeBatch
from components.standarize_stream import RunningMeanStd
from .ppo_learner import PPOLearner
import torch as th
from envs.graph_env import get_erdos_renyi_graph


class PPOGraphLearner(PPOLearner):
    def __init__(self, mac, scheme, logger, args):
        super().__init__(mac, scheme, logger, args)
        assert (
            not self.args.common_reward
        ), "Graph should not be used with common reward"
        if self.args.standardise_rewards:
            rew_shape = (1,) 
            self.rew_ms = RunningMeanStd(shape=rew_shape, device=self.device)
        self.graph_skip_ts = int(self.args.graph_skip_ts)
        if args.density is not None:
            assert args.density <= 1 and args.density >= 0
            args.k = args.density * self.n_agents
        self.use_full_path = args.use_full_path
        self.use_er_graph = args.use_er_graph
        self.use_knn_graph = args.use_knn_graph  # oracle graph
        assert not (self.use_er_graph and self.use_knn_graph), "pick one, or none"
        self.use_mi_graph = not self.use_knn_graph and not self.use_er_graph

        from modules.world.reverse_model import GraphModel, StateEncoder
        self.graph_model = GraphModel(scheme, args).to(self.device)
        self.state_encoder = StateEncoder(scheme, args).to(self.device)

        self.model_optim = th.optim.Adam(params=self.graph_model.parameters(), 
                                         lr=args.lr, 
                                         weight_decay=1e-3,
                                         decoupled_weight_decay=True)
        self.state_optim = th.optim.Adam(params=self.state_encoder.parameters(), 
                                         lr=args.lr, 
                                         weight_decay=1e-3,
                                         decoupled_weight_decay=True)

    def infer_graph(self, batch: EpisodeBatch, t_ep=None, **kwargs):
        if t_ep is not None:
            obs = batch['obs'][:, t_ep-1:t_ep+1]
        else:
            obs = batch["obs"]  # [thread, episode leng, n_agent, dim]
        thread, eps_len, n_agent, dim = obs.shape

        with th.no_grad():
            marginal_entropy, conditional_entropy = self.graph_model.fullbatch_forward(obs, self.state_encoder, batch.device)
            marginal_entropy = marginal_entropy.unsqueeze(-2).expand(-1, -1, n_agent, -1)
        assert marginal_entropy.shape == conditional_entropy.shape, f"{marginal_entropy.shape} {conditional_entropy.shape}"

        ratio = conditional_entropy / (marginal_entropy + 1e-8)
        graph = th.logical_and(ratio < 0.9, marginal_entropy > (np.log(self.n_actions) * 0.1))
        assert graph.shape[-1] == graph.shape[-2], graph.shape
        graph.diagonal(dim1=-2, dim2=-1).fill_(1)

        return graph, ratio, conditional_entropy, marginal_entropy

    def train_state_encoder(self, batch: EpisodeBatch, t_env):
        obs = batch["obs"]  # [thread, episode leng, n_agent, dim]
        action = batch["actions"]
        state_encoder = self.state_encoder
        state = state_encoder(obs)

        loss = self.state_encoder.single_agent_reverse_model.revrs_loss(state, action)
        return loss

    def train_reverse_model(self, batch: EpisodeBatch):
        obs = batch["obs"]  # [thread, episode leng, n_agent, dim]
        action = batch["actions"]
        state_encoder = self.state_encoder
        state = state_encoder(obs).detach()

        loss = self.graph_model.multi_agent_reverse_model.revrs_loss(state, action)
        loss = loss + self.graph_model.action_predictor.pred_loss(state, action)
        return loss

    def train_world_model(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        if not self.use_mi_graph:
            return
        action_prediction_loss = self.train_state_encoder(batch, t_env)
        self.state_optim.zero_grad()
        action_prediction_loss.backward()
        self.state_optim.step()

        reverse_model_loss = self.train_reverse_model(batch)
        self.model_optim.zero_grad()
        reverse_model_loss.backward()
        self.model_optim.step()

        self.logger.log_stat("reverse_model_loss", reverse_model_loss.item(), t_env)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities

        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :]

        terminated = batch["terminated"].float()
        mask = batch["filled"].float()
        assert mask.shape == terminated.shape
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = actions[:, :-1]

        if self.args.standardise_rewards:
            assert rewards.shape[-1] == self.rew_ms.mean.shape[-1], (
                f"{rewards.shape} {self.rew_ms.mean.shape}"
            ) 
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean.reshape([1]*len(rewards.shape))) / th.sqrt(self.rew_ms.var).reshape([1]*len(rewards.shape))

        if self.args.common_reward:
            assert (
                rewards.size(2) == 1
            ), "Expected singular agent dimension for common rewards"
            # reshape rewards to be of shape (batch_size, episode_length, n_agents)
            rewards = rewards.expand(-1, -1, self.n_agents)

        mask = mask.repeat(1, 1, self.n_agents)
        truncate_mask = mask[:, :-1]

        # critic_mask = truncate_mask.clone()

        old_mac_out = []
        self.old_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length - 1):
            agent_outs = self.old_mac.forward(batch, t=t)
            old_mac_out.append(agent_outs)
        old_mac_out = th.stack(old_mac_out, dim=1)  # Concat over time
        old_pi = old_mac_out
        old_pi[truncate_mask == 0] = 1.0

        old_entropy = -th.sum(old_pi * th.log(old_pi + 1e-10), dim=-1)

        old_pi_taken = th.gather(old_pi, dim=3, index=actions).squeeze(3)
        old_log_pi_taken = th.log(old_pi_taken + 1e-10)

        with th.no_grad():
            # graph = batch["graph"][:, :-1]
            if self.use_er_graph:
                batch_size, batch_timestep, n_agent = rewards.shape
                graph = get_erdos_renyi_graph(batch_size * batch_timestep, n_agent, p=self.args.p)
                graph = graph.reshape(batch_size, batch_timestep, n_agent, n_agent)
                graph = th.from_numpy(graph).to(self.device)
                graph.diagonal(dim1=-2, dim2=-1).fill_(1)
            elif self.use_mi_graph:
                graph, ratio, conditional_entropy, marginal_entropy = self.infer_graph(
                    batch
                )
            else:  # use oracle graph
                graph = batch["graph"][:, :-1]
            old_vals = self.critic(batch).squeeze(3)
            if not self.use_full_path:
                raise Exception("do not use this")
                advantages = utils.compute_adv_gae_graph(
                    self.args.gamma,
                    rewards,
                    mask,
                    old_vals,
                    0.95,
                    graph,
                    skip_ts=1,
                )
            else:
                # advantages = utils.compute_adv_gae_graph_fullpath(
                advantages = utils.compute_adv_gae_graph_naive(
                    self.args.gamma,
                    rewards,
                    mask,
                    old_vals,
                    0.95,
                    graph,
                    skip_ts=1,
                )

            target_vals = utils.compute_adv_gae(self.args.gamma, rewards, mask, old_vals, 0.95) + old_vals[:, :-1]

        for k in range(self.args.epochs):
            mac_out = []
            self.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length - 1):
                agent_outs = self.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)  # Concat over time

            pi = mac_out
            critic_train_stats = self.train_critic_sequential(
                self.critic, self.target_critic, batch, rewards, mask.clone(), target_vals
            )

            advantages = advantages.detach()
            # Calculate policy grad with mask

            pi[truncate_mask == 0] = 1.0

            pi_taken = th.gather(pi, dim=3, index=actions).squeeze(3)
            log_pi_taken = th.log(pi_taken + 1e-10)

            ratios = th.exp(log_pi_taken - old_log_pi_taken.detach())
            surr1 = ratios * advantages
            surr2 = (
                th.clamp(ratios, 1 - self.args.eps_clip, 1 + self.args.eps_clip)
                * advantages
            )

            entropy = -th.sum(pi * th.log(pi + 1e-10), dim=-1)
            pg_loss = (
                -(
                    (th.min(surr1, surr2) + self.args.entropy_coef * entropy)
                    * truncate_mask
                ).sum()
                / truncate_mask.sum()
            )

            # Optimise agents
            self.agent_optimiser.zero_grad()
            pg_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(
                self.agent_params, self.args.grad_norm_clip
            )
            self.agent_optimiser.step()

        self.old_mac.load_state(self.mac)

        # train reverse model
        # self.train_world_model(batch, t_env, episode_num)
        # for _ in range(1):
        #     model_loss = self.train_reverse_model(batch)
        # self.train_state_encoder(batch)

        # self.critic_training_steps += 1
        # if (
        #     self.args.target_update_interval_or_tau > 1
        #     and (self.critic_training_steps - self.last_target_update_step)
        #     / self.args.target_update_interval_or_tau
        #     >= 1.0
        # ):
        #     self._update_targets_hard()
        #     self.last_target_update_step = self.critic_training_steps
        # elif self.args.target_update_interval_or_tau <= 1.0:
        #     self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            ts_logged = len(critic_train_stats["critic_loss"])
            for key in [
                "critic_loss",
                "critic_grad_norm",
                "td_error_abs",
                "q_taken_mean",
                "target_mean",
            ]:
                self.logger.log_stat(
                    key, sum(critic_train_stats[key]) / ts_logged, t_env
                )

            self.logger.log_stat(
                "advantage_mean",
                (advantages * truncate_mask).sum().item() / truncate_mask.sum().item(),
                t_env,
            )
            self.logger.log_stat("pg_loss", pg_loss.item(), t_env)
            self.logger.log_stat("agent_grad_norm", grad_norm.item(), t_env)
            self.logger.log_stat(
                "pi_max",
                (pi.max(dim=-1)[0] * truncate_mask).sum().item()
                / truncate_mask.sum().item(),
                t_env,
            )

            # self.logger.log_stat("reverse_model_loss",  model_loss, t_env)
            self.logger.log_stat("entropy", entropy.mean().item(), t_env)
            self.logger.log_stat(
                "graph", graph.float().mean().item() * self.n_agents, t_env
            )
            if self.use_mi_graph:
                self.logger.log_stat("ratio", ratio.mean().item(), t_env)
                self.logger.log_stat("ratio_min", ratio.min().item(), t_env)
                self.logger.log_stat(
                    "marginal_entropy", marginal_entropy.mean().item(), t_env
                )
                self.logger.log_stat(
                    "conditional_entropy", conditional_entropy.mean().item(), t_env
                )
                self.logger.log_stat(
                    "conditional_entropy_var", conditional_entropy.var().item(), t_env
                )
            self.log_stats_t = t_env

    def train_critic_sequential(self, critic, target_critic, batch, rewards, mask, target_returns):

        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(
                self.ret_ms.var
            )
        mask = mask[:, :-1]

        running_log = {
            "critic_loss": [],
            "critic_grad_norm": [],
            "td_error_abs": [],
            "target_mean": [],
            "q_taken_mean": [],
        }

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = target_returns.detach() - v
        masked_td_error = td_error * mask
        loss = (masked_td_error**2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(
            self.critic_params, self.args.grad_norm_clip
        )
        self.critic_optimiser.step()

        running_log["critic_loss"].append(loss.item())
        running_log["critic_grad_norm"].append(grad_norm.item())
        mask_elems = mask.sum().item()
        running_log["td_error_abs"].append(
            (masked_td_error.abs().sum().item() / mask_elems)
        )
        running_log["q_taken_mean"].append((v * mask).sum().item() / mask_elems)
        running_log["target_mean"].append(
            (target_returns * mask).sum().item() / mask_elems
        )

        return running_log
