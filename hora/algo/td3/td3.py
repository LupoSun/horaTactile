# --------------------------------------------------------
# Minimal TD3 baseline for HORA Stage 1 oracle-policy experiments.
# --------------------------------------------------------

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from hora.algo.models.models import ActorCritic, MLP, PointNetEncoder
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.misc import AverageScalarMeter
from hora.utils.wandb_utils import init_wandb_run


class TD3ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, priv_dim, device, point_cloud_shape=None):
        self.capacity = int(capacity)
        self.device = device
        self.point_cloud_shape = point_cloud_shape
        self.ptr = 0
        self.size = 0
        self.obs = torch.zeros((self.capacity, obs_dim), dtype=torch.float32, device=device)
        self.next_obs = torch.zeros_like(self.obs)
        self.actions = torch.zeros((self.capacity, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.dones = torch.zeros((self.capacity, 1), dtype=torch.float32, device=device)
        self.priv_info = torch.zeros((self.capacity, priv_dim), dtype=torch.float32, device=device)
        self.next_priv_info = torch.zeros_like(self.priv_info)
        self.point_cloud = None
        self.next_point_cloud = None
        if point_cloud_shape is not None:
            self.point_cloud = torch.zeros((self.capacity, *point_cloud_shape), dtype=torch.float32, device=device)
            self.next_point_cloud = torch.zeros_like(self.point_cloud)

    def add_batch(self, obs_dict, actions, rewards, dones, next_obs_dict):
        batch_size = actions.shape[0]
        ids = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity
        self.obs[ids] = obs_dict["obs"]
        self.next_obs[ids] = next_obs_dict["obs"]
        self.actions[ids] = actions
        self.rewards[ids] = rewards.view(-1, 1)
        self.dones[ids] = dones.float().view(-1, 1)
        self.priv_info[ids] = obs_dict["priv_info"]
        self.next_priv_info[ids] = next_obs_dict["priv_info"]
        if self.point_cloud is not None:
            self.point_cloud[ids] = obs_dict["point_cloud"]
            self.next_point_cloud[ids] = next_obs_dict["point_cloud"]
        self.ptr = int((self.ptr + batch_size) % self.capacity)
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        ids = torch.randint(0, self.size, (batch_size,), device=self.device)
        batch = {
            "obs": self.obs[ids],
            "next_obs": self.next_obs[ids],
            "actions": self.actions[ids],
            "rewards": self.rewards[ids],
            "dones": self.dones[ids],
            "priv_info": self.priv_info[ids],
            "next_priv_info": self.next_priv_info[ids],
        }
        if self.point_cloud is not None:
            batch["point_cloud"] = self.point_cloud[ids]
            batch["next_point_cloud"] = self.next_point_cloud[ids]
        return batch


class TD3Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, units, priv_mlp_units, priv_info, priv_info_dim, use_shape_priv_info=False, pointnet_units=None):
        super(TD3Critic, self).__init__()
        self.priv_info = priv_info
        self.use_shape_priv_info = use_shape_priv_info
        self.priv_embed_dim = priv_mlp_units[-1] if priv_info else 0
        self.shape_embed_dim = pointnet_units[-1] if (use_shape_priv_info and pointnet_units) else 32
        extrin_dim = self.priv_embed_dim + (self.shape_embed_dim if use_shape_priv_info else 0)
        input_dim = obs_dim + act_dim + (extrin_dim if priv_info else 0)
        if priv_info:
            self.env_mlp = MLP(units=priv_mlp_units, input_size=priv_info_dim, activation=nn.ReLU)
            if use_shape_priv_info:
                self.pointnet = PointNetEncoder(units=pointnet_units)
        self.q1 = nn.Sequential(MLP(units=units, input_size=input_dim), nn.Linear(units[-1], 1))
        self.q2 = nn.Sequential(MLP(units=units, input_size=input_dim), nn.Linear(units[-1], 1))

    def encode_privileged(self, obs_dict):
        if not self.priv_info:
            return None
        phys_embedding = self.env_mlp(obs_dict["priv_info"])
        if not self.use_shape_priv_info:
            return torch.tanh(phys_embedding)
        shape_embedding = self.pointnet(obs_dict["point_cloud"])
        return torch.tanh(torch.cat([phys_embedding, shape_embedding], dim=-1))

    def forward(self, obs, action, obs_dict):
        features = [obs, action]
        extrin = self.encode_privileged(obs_dict)
        if extrin is not None:
            features.append(extrin)
        x = torch.cat(features, dim=-1)
        return self.q1(x), self.q2(x)

    def q1_value(self, obs, action, obs_dict):
        features = [obs, action]
        extrin = self.encode_privileged(obs_dict)
        if extrin is not None:
            features.append(extrin)
        return self.q1(torch.cat(features, dim=-1))


class TD3(object):
    def __init__(self, env, output_dif, full_config):
        self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.obs_shape = self.env.observation_space.shape
        self.priv_info = self.ppo_config["priv_info"]
        self.priv_info_dim = self.ppo_config["priv_info_dim"]
        self.use_shape_priv_info = self.ppo_config.get("use_shape_priv_info", False)
        point_cloud_shape = (self.ppo_config.get("n_pointcloud_pts", 100), 3) if self.use_shape_priv_info else None

        net_config = {
            "actor_units": self.network_config.mlp.units,
            "priv_mlp_units": self.network_config.priv_mlp.units,
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": self.priv_info,
            "proprio_adapt": False,
            "priv_info_dim": self.priv_info_dim,
            "use_shape_priv_info": self.use_shape_priv_info,
            "shape_embed_dim": self.ppo_config.get("shape_embed_dim", 32),
            "pointnet_units": self.ppo_config.get("pointnet_units", [32, 32, 32]),
            "asymmetric_critic": self.ppo_config.get("asymmetric_critic", False),
            "actor_use_privileged_info": self.ppo_config.get("actor_use_privileged_info", True),
            "recurrent_obs": self.ppo_config.get("recurrent_obs", False),
            "recurrent_obs_seq_len": self.ppo_config.get("recurrent_obs_seq_len", 3),
            "recurrent_hidden_size": self.ppo_config.get("recurrent_hidden_size", 128),
            "contact_event_gating": self.ppo_config.get("contact_event_gating", False),
            "contact_num_modes": self.ppo_config.get("contact_num_modes", 4),
            "contact_gate_hidden_size": self.ppo_config.get("contact_gate_hidden_size", 32),
            "contact_tactile_dim": self.ppo_config.get("contact_tactile_dim", 12),
            "contact_history_len": self.ppo_config.get("contact_history_len", 3),
        }
        self.actor = ActorCritic(net_config).to(self.device)
        self.actor_target = ActorCritic(net_config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = TD3Critic(
            self.obs_shape[0],
            self.actions_num,
            self.network_config.mlp.units,
            self.network_config.priv_mlp.units,
            self.priv_info,
            self.priv_info_dim,
            use_shape_priv_info=self.use_shape_priv_info,
            pointnet_units=self.ppo_config.get("pointnet_units", [32, 32, 32]),
        ).to(self.device)
        self.critic_target = TD3Critic(
            self.obs_shape[0],
            self.actions_num,
            self.network_config.mlp.units,
            self.network_config.priv_mlp.units,
            self.priv_info,
            self.priv_info_dim,
            use_shape_priv_info=self.use_shape_priv_info,
            pointnet_units=self.ppo_config.get("pointnet_units", [32, 32, 32]),
        ).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, self.ppo_config.get("nn_dir", "stage1_nn"))
        os.makedirs(self.nn_dir, exist_ok=True)
        init_wandb_run(full_config, name=self.ppo_config["output_name"], group=self.ppo_config.get("wandb_group") or "stage1_td3")

        self.gamma = self.ppo_config.get("gamma", 0.99)
        self.tau = self.ppo_config.get("td3_target_tau", 0.005)
        self.batch_size = self.ppo_config.get("td3_batch_size", self.ppo_config.get("minibatch_size", 32768))
        self.replay_size = self.ppo_config.get("td3_replay_size", 100000)
        self.learning_starts = self.ppo_config.get("td3_learning_starts", self.num_actors * 4)
        self.policy_delay = self.ppo_config.get("td3_policy_delay", 2)
        self.exploration_noise = self.ppo_config.get("td3_exploration_noise", 0.1)
        self.target_noise = self.ppo_config.get("td3_target_noise", 0.2)
        self.target_noise_clip = self.ppo_config.get("td3_target_noise_clip", 0.5)
        self.reward_scale = self.ppo_config.get("td3_reward_scale", 0.01)
        self.max_agent_steps = self.ppo_config["max_agent_steps"]
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.ppo_config["learning_rate"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.ppo_config["learning_rate"])
        self.replay = TD3ReplayBuffer(
            self.replay_size,
            self.obs_shape[0],
            self.actions_num,
            self.priv_info_dim,
            self.device,
            point_cloud_shape=point_cloud_shape,
        )
        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.current_rewards = torch.zeros((self.num_actors, 1), dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.agent_steps = 0
        self.gradient_steps = 0
        self.best_rewards = -10000

    @torch.no_grad()
    def _act(self, obs_dict, noise=True):
        processed_obs = self.running_mean_std(obs_dict["obs"])
        input_dict = {"obs": processed_obs, "priv_info": obs_dict["priv_info"]}
        if self.use_shape_priv_info:
            input_dict["point_cloud"] = obs_dict["point_cloud"]
        actions = self.actor.act_inference(input_dict)
        if noise:
            actions = actions + torch.randn_like(actions) * self.exploration_noise
        return torch.clamp(actions, -1.0, 1.0)

    def train(self):
        obs = self.env.reset()
        start_time = time.time()
        while self.agent_steps < self.max_agent_steps:
            self.running_mean_std.train()
            _ = self.running_mean_std(obs["obs"])
            self.running_mean_std.eval()
            if self.agent_steps < self.learning_starts:
                actions = torch.empty((self.num_actors, self.actions_num), device=self.device).uniform_(-1.0, 1.0)
            else:
                actions = self._act(obs, noise=True)
            next_obs, rewards, dones, infos = self.env.step(actions)
            self.replay.add_batch(obs, actions, rewards * self.reward_scale, dones, next_obs)
            self.agent_steps += self.num_actors
            self._track_episodes(rewards, dones, infos)
            obs = next_obs
            if self.replay.size >= max(self.batch_size, self.learning_starts):
                self._update()
            if self.agent_steps % max(self.num_actors * 10, 1) == 0:
                self._log(start_time)
        print("max steps achieved")
        wandb.finish()

    def _track_episodes(self, rewards, dones, infos):
        self.current_rewards += rewards.unsqueeze(1)
        self.current_lengths += 1
        done_indices = dones.nonzero(as_tuple=False)
        self.episode_rewards.update(self.current_rewards[done_indices])
        self.episode_lengths.update(self.current_lengths[done_indices])
        not_dones = 1.0 - dones.float()
        self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
        self.current_lengths = self.current_lengths * not_dones

    def _update(self):
        batch = self.replay.sample(self.batch_size)
        obs = self.running_mean_std(batch["obs"])
        next_obs = self.running_mean_std(batch["next_obs"])
        obs_dict = {"priv_info": batch["priv_info"]}
        next_obs_dict = {"priv_info": batch["next_priv_info"]}
        if self.use_shape_priv_info:
            obs_dict["point_cloud"] = batch["point_cloud"]
            next_obs_dict["point_cloud"] = batch["next_point_cloud"]
        with torch.no_grad():
            target_actor_input = {"obs": next_obs, "priv_info": batch["next_priv_info"]}
            if self.use_shape_priv_info:
                target_actor_input["point_cloud"] = batch["next_point_cloud"]
            next_action = self.actor_target.act_inference(target_actor_input)
            noise = torch.randn_like(next_action) * self.target_noise
            noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            target_q1, target_q2 = self.critic_target(next_obs, next_action, next_obs_dict)
            target_q = batch["rewards"] + (1.0 - batch["dones"]) * self.gamma * torch.min(target_q1, target_q2)

        current_q1, current_q2 = self.critic(obs, batch["actions"], obs_dict)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = torch.zeros((), device=self.device)
        if self.gradient_steps % self.policy_delay == 0:
            actor_input = {"obs": obs, "priv_info": batch["priv_info"]}
            if self.use_shape_priv_info:
                actor_input["point_cloud"] = batch["point_cloud"]
            actor_action = self.actor._actor_critic(actor_input)[0]
            actor_loss = -self.critic.q1_value(obs, actor_action, obs_dict).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)

        self.gradient_steps += 1
        self.last_actor_loss = actor_loss.detach()
        self.last_critic_loss = critic_loss.detach()

    def _soft_update(self, source, target):
        for src_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.mul_(1.0 - self.tau).add_(src_param.data, alpha=self.tau)

    def _log(self, start_time):
        mean_rewards = self.episode_rewards.get_mean()
        mean_lengths = self.episode_lengths.get_mean()
        fps = self.agent_steps / max(time.time() - start_time, 1e-6)
        print(f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {fps:.1f} | Current Best: {self.best_rewards:.2f}")
        wandb.log(
            {
                "performance/EnvStepFPS": fps,
                "losses/actor_loss": float(getattr(self, "last_actor_loss", torch.zeros(())).item()),
                "losses/critic_loss": float(getattr(self, "last_critic_loss", torch.zeros(())).item()),
                "episode_rewards/step": mean_rewards,
                "episode_lengths/step": mean_lengths,
            },
            step=self.agent_steps,
        )
        if mean_rewards > self.best_rewards:
            print(f"save current best reward: {mean_rewards:.2f}")
            self.best_rewards = mean_rewards
            self.save(os.path.join(self.nn_dir, "best"))

    def save(self, name):
        weights = {
            "model": self.actor.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "running_mean_std": self.running_mean_std.state_dict(),
        }
        torch.save(weights, f"{name}.pth")

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor_target.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic_target.load_state_dict(checkpoint["critic"])
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.actor.load_state_dict(checkpoint["actor"])
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])

    def test(self):
        self.actor.eval()
        self.running_mean_std.eval()
        obs = self.env.reset()
        while True:
            actions = self._act(obs, noise=False)
            obs, _, _, _ = self.env.step(actions)
