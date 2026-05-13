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
    def __init__(
        self,
        input_size,
        recurrent=False,
        seq_len=3,
        hidden_size=128,
        contact_reset_recurrent=False,
        contact_tactile_dim=12,
        contact_gate_hidden_size=32,
    ):
        super(ObservationEncoder, self).__init__()
        self.input_size = input_size
        self.recurrent = recurrent
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.contact_reset_recurrent = contact_reset_recurrent
        self.contact_tactile_dim = contact_tactile_dim
        if recurrent:
            if contact_reset_recurrent:
                tactile_history_dim = seq_len * contact_tactile_dim
                proprio_history_dim = input_size - tactile_history_dim
                if proprio_history_dim <= 0 or proprio_history_dim % seq_len != 0:
                    raise ValueError(
                        f"Contact-reset recurrence needs obs dim {input_size} to contain "
                        f"{seq_len} tactile frames of dim {contact_tactile_dim}"
                    )
                self.proprio_frame_dim = proprio_history_dim // seq_len
                self.frame_dim = self.proprio_frame_dim + contact_tactile_dim
                self.gru_cell = nn.GRUCell(self.frame_dim, hidden_size)
                self.reset_gate = nn.Sequential(
                    nn.Linear(contact_tactile_dim * 2, contact_gate_hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(contact_gate_hidden_size, 1),
                    nn.Sigmoid(),
                )
            else:
                if input_size % seq_len != 0:
                    raise ValueError(f"Cannot split observation dim {input_size} into {seq_len} recurrent frames")
                self.frame_dim = input_size // seq_len
                self.gru = nn.GRU(self.frame_dim, hidden_size, batch_first=True)
            self.output_size = hidden_size
        else:
            self.frame_dim = input_size
            self.output_size = input_size

    def _contact_reset_frames(self, obs):
        proprio_end = self.seq_len * self.proprio_frame_dim
        proprio = obs[:, :proprio_end].view(obs.shape[0], self.seq_len, self.proprio_frame_dim)
        tactile = obs[:, proprio_end:].view(obs.shape[0], self.seq_len, self.contact_tactile_dim)
        return torch.cat([proprio, tactile], dim=-1), tactile

    def forward(self, obs):
        if not self.recurrent:
            return obs
        if self.contact_reset_recurrent:
            obs_seq, tactile_seq = self._contact_reset_frames(obs)
            hidden = obs.new_zeros((obs.shape[0], self.hidden_size))
            prev_tactile = tactile_seq[:, 0]
            for idx in range(self.seq_len):
                if idx > 0:
                    current_tactile = tactile_seq[:, idx]
                    event = (current_tactile - prev_tactile).abs()
                    reset = self.reset_gate(torch.cat([current_tactile.abs(), event], dim=-1))
                    hidden = hidden * (1.0 - reset)
                    prev_tactile = current_tactile
                hidden = self.gru_cell(obs_seq[:, idx], hidden)
            return hidden
        obs_seq = obs.view(obs.shape[0], self.seq_len, self.frame_dim)
        _, hidden = self.gru(obs_seq)
        return hidden[-1]


class ContactEventGate(nn.Module):
    def __init__(
        self,
        obs_dim,
        history_len=3,
        tactile_dim=12,
        hidden_size=32,
        num_modes=4,
        event_features=False,
        contact_threshold=0.05,
    ):
        super(ContactEventGate, self).__init__()
        self.history_len = history_len
        self.tactile_dim = tactile_dim
        self.num_modes = num_modes
        self.event_features = event_features
        self.contact_threshold = contact_threshold
        self.enabled = obs_dim >= history_len * tactile_dim
        if not self.enabled:
            input_size = 1
        elif event_features:
            input_size = tactile_dim * 9 + 3
        else:
            input_size = tactile_dim * 3
        self.feature_dim = input_size
        self.gate = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_modes),
        )

    def _tactile_history(self, obs):
        tactile = obs[:, -self.history_len * self.tactile_dim:]
        return tactile.view(obs.shape[0], self.history_len, self.tactile_dim)

    def _contact_features(self, obs, frame_index=None):
        if not self.enabled:
            return obs.new_zeros((obs.shape[0], 1))
        tactile = self._tactile_history(obs)
        idx = self.history_len - 1 if frame_index is None else frame_index
        idx = max(0, min(idx, self.history_len - 1))
        previous_idx = max(idx - 1, 0)
        current = tactile[:, idx].abs()
        previous = tactile[:, previous_idx].abs()
        event = (current - previous).abs()
        context = tactile[:, :idx + 1].abs().mean(dim=1)
        if not self.event_features:
            return torch.cat([current, event, context], dim=-1)

        signed_delta = current - previous
        current_contact = current > self.contact_threshold
        previous_contact = previous > self.contact_threshold
        contact_make = torch.logical_and(current_contact, torch.logical_not(previous_contact)).float()
        contact_break = torch.logical_and(torch.logical_not(current_contact), previous_contact).float()
        contact_now = current_contact.float()
        contact_duration = (tactile[:, :idx + 1].abs() > self.contact_threshold).float().mean(dim=1)
        scalar_counts = torch.cat(
            [
                contact_now.mean(dim=1, keepdim=True),
                contact_make.mean(dim=1, keepdim=True),
                contact_break.mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                current,
                previous,
                signed_delta,
                event,
                context,
                contact_now,
                contact_make,
                contact_break,
                contact_duration,
                scalar_counts,
            ],
            dim=-1,
        )

    def forward(self, obs):
        return torch.softmax(self.gate(self._contact_features(obs)), dim=-1)

    def forward_with_stats(self, obs):
        gates = self.forward(obs)
        mean_gates = gates.mean(dim=0)
        uniform = torch.full_like(mean_gates, 1.0 / self.num_modes)
        balance_loss = torch.sum((mean_gates - uniform) ** 2)
        entropy = -(gates * torch.log(gates.clamp_min(1e-8))).sum(dim=-1).mean()
        max_prob = gates.max(dim=-1).values.mean()
        if self.enabled and self.history_len > 1:
            prev_gates = torch.softmax(
                self.gate(self._contact_features(obs, frame_index=self.history_len - 2)),
                dim=-1,
            )
            switch_loss = torch.mean(torch.sum((gates - prev_gates) ** 2, dim=-1))
            tactile = self._tactile_history(obs)
            current = tactile[:, -1].abs() > self.contact_threshold
            previous = tactile[:, -2].abs() > self.contact_threshold
            make = torch.logical_and(current, torch.logical_not(previous)).float().mean()
            contact_break = torch.logical_and(torch.logical_not(current), previous).float().mean()
            active = current.float().mean()
            event = torch.logical_xor(current, previous).float().mean()
        else:
            switch_loss = gates.new_zeros(())
            make = gates.new_zeros(())
            contact_break = gates.new_zeros(())
            active = gates.new_zeros(())
            event = gates.new_zeros(())
        return gates, {
            'mean_gates': mean_gates,
            'entropy': entropy,
            'max_prob': max_prob,
            'balance_loss': balance_loss,
            'switch_loss': switch_loss,
            'contact_make_rate': make,
            'contact_break_rate': contact_break,
            'active_contact_rate': active,
            'contact_event_rate': event,
        }


class ContactOptionController(nn.Module):
    def __init__(
        self,
        obs_dim,
        actor_feature_dim,
        history_len=3,
        tactile_dim=12,
        hidden_size=32,
        num_modes=4,
        event_features=True,
        contact_threshold=0.05,
    ):
        super(ContactOptionController, self).__init__()
        self.num_modes = num_modes
        self.feature_extractor = ContactEventGate(
            obs_dim,
            history_len=history_len,
            tactile_dim=tactile_dim,
            hidden_size=hidden_size,
            num_modes=num_modes,
            event_features=event_features,
            contact_threshold=contact_threshold,
        )
        manager_input_dim = actor_feature_dim + self.feature_extractor.feature_dim
        self.option_policy = nn.Sequential(
            nn.Linear(manager_input_dim, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_modes),
        )
        self.termination_policy = nn.Sequential(
            nn.Linear(manager_input_dim + num_modes, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 1),
        )

    def manager_features(self, actor_features, obs):
        contact_features = self.feature_extractor._contact_features(obs)
        return torch.cat([actor_features, contact_features], dim=-1)

    def forward(self, actor_features, obs, prev_options):
        manager_features = self.manager_features(actor_features, obs)
        option_logits = self.option_policy(manager_features)
        prev_one_hot = torch.nn.functional.one_hot(prev_options.long(), self.num_modes).float()
        termination_logits = self.termination_policy(torch.cat([manager_features, prev_one_hot], dim=-1)).squeeze(-1)
        return option_logits, termination_logits


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
        self.contact_reset_recurrent = kwargs.get('contact_reset_recurrent', False)
        self.contact_options = kwargs.get('contact_options', False)
        self.contact_event_gating = kwargs.get('contact_event_gating', False)
        self.contact_tactile_dim = kwargs.get('contact_tactile_dim', 12)
        self.contact_history_len = kwargs.get('contact_history_len', self.recurrent_obs_seq_len)
        self.contact_num_modes = kwargs.get('contact_num_modes', 4)
        self.contact_gate_hidden_size = kwargs.get('contact_gate_hidden_size', 32)
        self.contact_gate_event_features = kwargs.get('contact_gate_event_features', False)
        self.contact_gate_threshold = kwargs.get('contact_gate_threshold', 0.05)
        self.contact_transition_aux_loss = kwargs.get('contact_transition_aux_loss', False)
        self.actor_obs_encoder = ObservationEncoder(
            obs_input_shape,
            recurrent=self.recurrent_obs,
            seq_len=self.recurrent_obs_seq_len,
            hidden_size=self.recurrent_hidden_size,
            contact_reset_recurrent=self.contact_reset_recurrent,
            contact_tactile_dim=self.contact_tactile_dim,
            contact_gate_hidden_size=self.contact_gate_hidden_size,
        )
        self.has_separate_critic = self.asymmetric_critic or not self.actor_use_privileged_info
        self.critic_obs_encoder = None
        if self.asymmetric_critic:
            self.critic_obs_encoder = ObservationEncoder(
                obs_input_shape,
                recurrent=self.recurrent_obs,
                seq_len=self.recurrent_obs_seq_len,
                hidden_size=self.recurrent_hidden_size,
                contact_reset_recurrent=self.contact_reset_recurrent,
                contact_tactile_dim=self.contact_tactile_dim,
                contact_gate_hidden_size=self.contact_gate_hidden_size,
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
        if self.contact_options:
            self.contact_option_controller = ContactOptionController(
                obs_input_shape,
                actor_feature_dim=out_size,
                history_len=self.contact_history_len,
                tactile_dim=self.contact_tactile_dim,
                hidden_size=self.contact_gate_hidden_size,
                num_modes=self.contact_num_modes,
                event_features=self.contact_gate_event_features,
                contact_threshold=self.contact_gate_threshold,
            )
            self.mu_experts = nn.ModuleList(
                [torch.nn.Linear(out_size, actions_num) for _ in range(self.contact_num_modes)]
            )
            self.contact_gate = None
        elif self.contact_event_gating:
            self.contact_gate = ContactEventGate(
                obs_input_shape,
                history_len=self.contact_history_len,
                tactile_dim=self.contact_tactile_dim,
                hidden_size=self.contact_gate_hidden_size,
                num_modes=self.contact_num_modes,
                event_features=self.contact_gate_event_features,
                contact_threshold=self.contact_gate_threshold,
            )
            self.mu_experts = nn.ModuleList(
                [torch.nn.Linear(out_size, actions_num) for _ in range(self.contact_num_modes)]
            )
        else:
            self.contact_option_controller = None
            self.contact_gate = None
            self.mu = torch.nn.Linear(out_size, actions_num)
        if self.contact_transition_aux_loss:
            self.contact_transition_head = nn.Sequential(
                torch.nn.Linear(out_size, self.contact_gate_hidden_size),
                nn.ReLU(inplace=True),
                torch.nn.Linear(self.contact_gate_hidden_size, self.contact_tactile_dim),
            )
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
        option_state = obs_dict.get('contact_option_state')
        rst = self._actor_critic(obs_dict, option_state=option_state)
        mu, logstd, value = rst[:3]
        option_result = rst[5] if len(rst) > 5 else None
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
        if option_result is not None:
            result.update(option_result)
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        option_state = obs_dict.get('contact_option_state')
        rst = self._actor_critic(obs_dict, option_state=option_state)
        mu, logstd, value = rst[:3]
        option_result = rst[5] if len(rst) > 5 else None
        if option_result is not None:
            obs_dict['contact_option_result'] = option_result
        return mu

    def _option_mu(self, actor_features, option_ids):
        expert_mus = torch.stack([expert(actor_features) for expert in self.mu_experts], dim=1)
        gather_index = option_ids.long().view(-1, 1, 1).expand(-1, 1, expert_mus.shape[-1])
        return expert_mus.gather(1, gather_index).squeeze(1)

    def _sample_contact_options(self, actor_features, obs, option_state):
        batch_size = obs.shape[0]
        if option_state is None:
            prev_options = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
            option_dwell = torch.zeros(batch_size, dtype=torch.long, device=obs.device)
            reset_mask = torch.ones(batch_size, dtype=torch.bool, device=obs.device)
            force_switch_mask = torch.zeros(batch_size, dtype=torch.bool, device=obs.device)
        else:
            prev_options = option_state.get('prev_options', torch.zeros(batch_size, dtype=torch.long, device=obs.device)).long()
            option_dwell = option_state.get('option_dwell', torch.zeros(batch_size, dtype=torch.long, device=obs.device)).long()
            reset_mask = option_state.get('reset_mask', torch.zeros(batch_size, dtype=torch.bool, device=obs.device)).bool()
            force_switch_mask = option_state.get('force_switch_mask', torch.zeros(batch_size, dtype=torch.bool, device=obs.device)).bool()

        option_logits, termination_logits = self.contact_option_controller(actor_features, obs, prev_options)
        option_dist = torch.distributions.Categorical(logits=option_logits)
        termination_dist = torch.distributions.Bernoulli(logits=termination_logits)
        termination_mask = ~(reset_mask | force_switch_mask)
        if option_state is not None and 'option_ids' in option_state:
            option_ids = option_state['option_ids'].long()
            boundary = option_state.get('option_active', torch.ones(batch_size, device=obs.device)).bool()
            termination_targets = option_state.get(
                'termination_target',
                torch.zeros(batch_size, dtype=torch.float32, device=obs.device),
            ).float()
            termination_mask = option_state.get('termination_active', termination_mask.float()).bool()
        else:
            sampled_options = option_dist.sample()
            sampled_termination = termination_dist.sample().bool()
            boundary = reset_mask | force_switch_mask | (sampled_termination & termination_mask)
            option_ids = torch.where(boundary, sampled_options, prev_options)
            termination_targets = (sampled_termination & termination_mask).float()
        option_neglogp = -option_dist.log_prob(option_ids)
        termination_neglogp = -termination_dist.log_prob(termination_targets)
        return option_ids, {
            'contact_option_ids': option_ids,
            'contact_prev_option_ids': prev_options,
            'contact_option_neglogp': option_neglogp,
            'contact_option_active': boundary.float(),
            'contact_termination_neglogp': termination_neglogp,
            'contact_termination_target': termination_targets,
            'contact_termination_active': termination_mask.float(),
            'contact_option_dwell': option_dwell.float(),
            'contact_option_entropy': option_dist.entropy(),
            'contact_termination_entropy': termination_dist.entropy(),
            'contact_termination_prob': torch.sigmoid(termination_logits),
        }

    def _actor_mu(self, actor_features, obs, return_gate_stats=False, option_state=None):
        if self.contact_options:
            option_ids, option_result = self._sample_contact_options(actor_features, obs, option_state)
            return self._option_mu(actor_features, option_ids), option_result
        if not self.contact_event_gating:
            return (self.mu(actor_features), None) if return_gate_stats else self.mu(actor_features)
        if return_gate_stats:
            gates, gate_stats = self.contact_gate.forward_with_stats(obs)
        else:
            gates, gate_stats = self.contact_gate(obs), None
        expert_mus = torch.stack([expert(actor_features) for expert in self.mu_experts], dim=1)
        mu = torch.sum(expert_mus * gates.unsqueeze(-1), dim=1)
        return (mu, gate_stats) if return_gate_stats else mu

    def _actor_critic(
        self,
        obs_dict,
        return_contact_transition=False,
        return_contact_gate_stats=False,
        option_state=None,
    ):
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
        actor_mu_result = self._actor_mu(
            actor_features,
            obs,
            return_gate_stats=return_contact_gate_stats,
            option_state=option_state,
        )
        if self.contact_options:
            mu, contact_gate_stats = actor_mu_result
        elif return_contact_gate_stats:
            mu, contact_gate_stats = actor_mu_result
        else:
            mu, contact_gate_stats = actor_mu_result, None
        sigma = self.sigma
        contact_transition_logits = (
            self.contact_transition_head(actor_features)
            if self.contact_transition_aux_loss
            else None
        )
        if return_contact_gate_stats:
            return mu, mu * 0 + sigma, value, extrin, extrin_gt, contact_transition_logits, contact_gate_stats
        if return_contact_transition:
            return mu, mu * 0 + sigma, value, extrin, extrin_gt, contact_transition_logits
        if self.contact_options:
            return mu, mu * 0 + sigma, value, extrin, extrin_gt, contact_gate_stats
        return mu, mu * 0 + sigma, value, extrin, extrin_gt

    def _encode_privileged(self, obs_dict):
        phys_embedding = self.env_mlp(obs_dict['priv_info'])
        if not self.use_shape_priv_info:
            return phys_embedding
        shape_embedding = self.pointnet(obs_dict['point_cloud'])
        return torch.cat([phys_embedding, shape_embedding], dim=-1)

    def act_from_extrin(self, obs, extrin, option_state=None):
        actor_obs = self.actor_obs_encoder(obs)
        if self.actor_use_privileged_info:
            actor_obs = torch.cat([actor_obs, extrin], dim=-1)
        x = self.actor_mlp(actor_obs)
        mu_result = self._actor_mu(x, obs, option_state=option_state)
        return mu_result[0] if self.contact_options else mu_result

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(
            input_dict,
            return_contact_transition=True,
            return_contact_gate_stats=True,
            option_state=input_dict.get('contact_option_state'),
        )
        mu, logstd, value, extrin, extrin_gt, contact_transition_logits, contact_gate_stats = rst
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
        if contact_transition_logits is not None:
            result['contact_transition_logits'] = contact_transition_logits
        if contact_gate_stats is not None and self.contact_options:
            result['contact_option_stats'] = contact_gate_stats
        elif contact_gate_stats is not None:
            result['contact_gate_stats'] = contact_gate_stats
        return result
