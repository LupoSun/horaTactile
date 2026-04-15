import numpy as np
import torch
from gym import spaces
from omegaconf import OmegaConf

from hora.algo.padapt import padapt as padapt_module
from hora.algo.ppo import ppo as ppo_module
from hora.utils import wandb_utils


class DummyEnv:
    def __init__(self):
        self.action_space = spaces.Box(low=-np.ones(2, dtype=np.float32), high=np.ones(2, dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.ones(3, dtype=np.float32),
            high=np.ones(3, dtype=np.float32),
            dtype=np.float32,
        )
        self.prop_hist_len = 4
        self.hist_obs_dim = 32


class DummyModel(torch.nn.Module):
    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.adapt_tconv = torch.nn.Linear(1, 1)
        self.policy = torch.nn.Linear(1, 1)

    def to(self, _device):
        return self

    def eval(self):
        return self


class DummyRunningMeanStd:
    def __init__(self, *_args, **_kwargs):
        self.training = False

    def to(self, _device):
        return self

    def eval(self):
        self.training = False
        return self

    def train(self):
        self.training = True
        return self

    def state_dict(self):
        return {}


class DummyExperienceBuffer:
    def __init__(self, *_args, **_kwargs):
        pass


def make_full_config(proprio_adapt=False):
    return OmegaConf.create(
        {
            "rl_device": "cpu",
            "test": False,
            "train": {
                "algo": "ProprioAdapt" if proprio_adapt else "PPO",
                "network": {
                    "mlp": {"units": [8, 4]},
                    "priv_mlp": {"units": [4, 2]},
                },
                "ppo": {
                    "output_name": "AllegroHandHora/test_run",
                    "normalize_input": True,
                    "normalize_value": True,
                    "value_bootstrap": True,
                    "num_actors": 2,
                    "normalize_advantage": True,
                    "gamma": 0.99,
                    "tau": 0.95,
                    "learning_rate": 1e-3,
                    "kl_threshold": 0.02,
                    "horizon_length": 2,
                    "minibatch_size": 4,
                    "mini_epochs": 1,
                    "clip_value": True,
                    "critic_coef": 4,
                    "entropy_coef": 0.0,
                    "e_clip": 0.2,
                    "bounds_loss_coef": 0.0001,
                    "truncate_grads": True,
                    "grad_norm": 1.0,
                    "save_best_after": 0,
                    "save_frequency": 500,
                    "max_agent_steps": 1024,
                    "priv_info": True,
                    "priv_info_dim": 9,
                    "priv_info_embed_dim": 8,
                    "proprio_adapt": proprio_adapt,
                },
            },
        }
    )


def test_resolve_wandb_mode_prefers_env_override():
    assert wandb_utils.resolve_wandb_mode({}) == "offline"
    assert wandb_utils.resolve_wandb_mode({"WANDB_API_KEY": "secret"}) == "online"
    assert wandb_utils.resolve_wandb_mode({"WANDB_API_KEY": "secret", "WANDB_MODE": "disabled"}) == "disabled"


def test_init_wandb_run_passes_resolved_config(monkeypatch):
    init_kwargs = {}
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: init_kwargs.update(kwargs) or kwargs)

    config = OmegaConf.create({"train": {"ppo": {"output_name": "demo"}}})
    wandb_utils.init_wandb_run(config, name="demo", group="stage1")

    assert init_kwargs["project"] == "hora"
    assert init_kwargs["name"] == "demo"
    assert init_kwargs["group"] == "stage1"
    assert init_kwargs["mode"] == "offline"
    assert init_kwargs["config"] == {"train": {"ppo": {"output_name": "demo"}}}


def test_ppo_initializes_wandb_with_stage1_group(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(ppo_module, "ActorCritic", DummyModel)
    monkeypatch.setattr(ppo_module, "RunningMeanStd", DummyRunningMeanStd)
    monkeypatch.setattr(ppo_module, "ExperienceBuffer", DummyExperienceBuffer)
    monkeypatch.setattr(
        ppo_module,
        "init_wandb_run",
        lambda full_config, name, group: calls.append((full_config, name, group)),
    )

    ppo_module.PPO(DummyEnv(), str(tmp_path), make_full_config(proprio_adapt=False))

    assert len(calls) == 1
    _, name, group = calls[0]
    assert name == "AllegroHandHora/test_run"
    assert group == "stage1"


def test_proprio_adapt_initializes_wandb_with_stage2_group(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(padapt_module, "ActorCritic", DummyModel)
    monkeypatch.setattr(padapt_module, "RunningMeanStd", DummyRunningMeanStd)
    monkeypatch.setattr(
        padapt_module,
        "init_wandb_run",
        lambda full_config, name, group: calls.append((full_config, name, group)),
    )

    agent = padapt_module.ProprioAdapt(DummyEnv(), str(tmp_path), make_full_config(proprio_adapt=True))

    assert len(calls) == 1
    _, name, group = calls[0]
    assert name == "AllegroHandHora/test_run"
    assert group == "stage2"
    assert agent.max_agent_steps == 1024


def test_proprio_adapt_uses_env_hist_obs_dim(monkeypatch, tmp_path):
    env = DummyEnv()
    env.hist_obs_dim = 44
    init_calls = []
    rms_inputs = []

    monkeypatch.setattr(
        padapt_module,
        "ActorCritic",
        lambda net_config: init_calls.append(net_config) or DummyModel(),
    )

    class CapturingRunningMeanStd(DummyRunningMeanStd):
        def __init__(self, insize, *_args, **_kwargs):
            super().__init__()
            rms_inputs.append(insize)

    monkeypatch.setattr(padapt_module, "RunningMeanStd", CapturingRunningMeanStd)
    monkeypatch.setattr(
        padapt_module,
        "init_wandb_run",
        lambda full_config, name, group: None,
    )

    padapt_module.ProprioAdapt(env, str(tmp_path), make_full_config(proprio_adapt=True))

    assert init_calls[0]["hist_obs_dim"] == 44
    assert (env.prop_hist_len, 44) in rms_inputs


def test_proprio_adapt_tconv_supports_tactile_history_width():
    model = padapt_module.ActorCritic(
        {
            "actor_units": [8, 4],
            "priv_mlp_units": [4, 8],
            "actions_num": 2,
            "input_shape": (3,),
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
            "hist_obs_dim": 44,
        }
    )

    mu, logstd, value, extrin, extrin_gt = model._actor_critic(
        {
            "obs": torch.randn(2, 3),
            "priv_info": torch.randn(2, 9),
            "proprio_hist": torch.randn(2, 30, 44),
        }
    )

    assert mu.shape == (2, 2)
    assert logstd.shape == (2, 2)
    assert value.shape == (2, 1)
    assert extrin.shape == (2, 8)
    assert extrin_gt.shape == (2, 8)
