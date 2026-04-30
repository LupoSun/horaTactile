# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, units, input_size, activation=nn.ELU):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(activation())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class ProprioAdaptTConv(nn.Module):
    def __init__(self, hist_obs_dim=32, output_dim=8):
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(hist_obs_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(32, 32, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(32 * 3, output_dim)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 50, 32)
        x = x.permute((0, 2, 1))  # (N, 32, 50)
        x = self.temporal_aggregation(x)  # (N, 32, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class PointNetEncoder(nn.Module):
    def __init__(self, units=None):
        super(PointNetEncoder, self).__init__()
        units = [32, 32, 32] if units is None else units
        layers = []
        input_size = 3
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU(inplace=True))
            input_size = output_size
        self.point_mlp = nn.Sequential(*layers)
        self.embed_dim = units[-1]

    def forward(self, point_cloud):
        point_features = self.point_mlp(point_cloud)
        return point_features.max(dim=1).values


class ObservationEncoder(nn.Module):
    def __init__(self, input_size, recurrent=False, seq_len=3, hidden_size=128):
        super(ObservationEncoder, self).__init__()
        self.input_size = input_size
        self.recurrent = recurrent
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        if recurrent:
            if input_size % seq_len != 0:
                raise ValueError(f"Cannot split observation dim {input_size} into {seq_len} recurrent frames")
            self.frame_dim = input_size // seq_len
            self.gru = nn.GRU(self.frame_dim, hidden_size, batch_first=True)
            self.output_size = hidden_size
        else:
            self.frame_dim = input_size
            self.output_size = input_size

    def forward(self, obs):
        if not self.recurrent:
            return obs
        obs_seq = obs.view(obs.shape[0], self.seq_len, self.frame_dim)
        _, hidden = self.gru(obs_seq)
        return hidden[-1]


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        obs_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        self.use_shape_priv_info = kwargs.get('use_shape_priv_info', False)
        self.shape_embed_dim = kwargs.get('shape_embed_dim', 32)
        self.pointnet_units = kwargs.get('pointnet_units', [32, 32, self.shape_embed_dim])
        self.asymmetric_critic = kwargs.get('asymmetric_critic', False)
        self.actor_use_privileged_info = kwargs.get('actor_use_privileged_info', True)
        self.recurrent_obs = kwargs.get('recurrent_obs', False)
        self.recurrent_obs_seq_len = kwargs.get('recurrent_obs_seq_len', 3)
        self.recurrent_hidden_size = kwargs.get('recurrent_hidden_size', 128)
        self.actor_obs_encoder = ObservationEncoder(
            obs_input_shape,
            recurrent=self.recurrent_obs,
            seq_len=self.recurrent_obs_seq_len,
            hidden_size=self.recurrent_hidden_size,
        )
        self.has_separate_critic = self.asymmetric_critic or not self.actor_use_privileged_info
        self.critic_obs_encoder = None
        if self.asymmetric_critic:
            self.critic_obs_encoder = ObservationEncoder(
                obs_input_shape,
                recurrent=self.recurrent_obs,
                seq_len=self.recurrent_obs_seq_len,
                hidden_size=self.recurrent_hidden_size,
            )
        actor_input_shape = self.actor_obs_encoder.output_size
        critic_input_shape = (
            self.critic_obs_encoder.output_size
            if self.critic_obs_encoder is not None
            else self.actor_obs_encoder.output_size
        )
        self.priv_embed_dim = self.priv_mlp[-1] if self.priv_info else 0
        self.extrin_dim = self.priv_embed_dim + (self.shape_embed_dim if self.use_shape_priv_info else 0)
        if self.priv_info:
            if self.actor_use_privileged_info:
                actor_input_shape += self.extrin_dim
            critic_input_shape += self.extrin_dim
            self.env_mlp = MLP(units=self.priv_mlp, input_size=kwargs['priv_info_dim'], activation=nn.ReLU)
            if self.use_shape_priv_info:
                self.pointnet = PointNetEncoder(units=self.pointnet_units)

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(kwargs.get('hist_obs_dim', 32), output_dim=self.extrin_dim)

        self.actor_mlp = MLP(units=self.units, input_size=actor_input_shape)
        self.critic_mlp = None
        if self.has_separate_critic:
            self.critic_mlp = MLP(units=self.units, input_size=critic_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma, 0)

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        actor_obs = self.actor_obs_encoder(obs)
        critic_obs = actor_obs
        if self.critic_obs_encoder is not None:
            critic_obs = self.critic_obs_encoder(obs)
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label
                extrin_gt = self._encode_privileged(obs_dict) if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
            else:
                extrin = self._encode_privileged(obs_dict)
                extrin = torch.tanh(extrin)
            if self.actor_use_privileged_info:
                actor_obs = torch.cat([actor_obs, extrin], dim=-1)
            critic_obs = torch.cat([critic_obs, extrin], dim=-1)

        actor_features = self.actor_mlp(actor_obs)
        critic_features = self.critic_mlp(critic_obs) if self.critic_mlp is not None else actor_features
        value = self.value(critic_features)
        mu = self.mu(actor_features)
        sigma = self.sigma
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def _encode_privileged(self, obs_dict):
        phys_embedding = self.env_mlp(obs_dict['priv_info'])
        if not self.use_shape_priv_info:
            return phys_embedding
        shape_embedding = self.pointnet(obs_dict['point_cloud'])
        return torch.cat([phys_embedding, shape_embedding], dim=-1)

    def act_from_extrin(self, obs, extrin):
        actor_obs = self.actor_obs_encoder(obs)
        if self.actor_use_privileged_info:
            actor_obs = torch.cat([actor_obs, extrin], dim=-1)
        x = self.actor_mlp(actor_obs)
        return self.mu(x)

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
        }
        return result
