import os
import argparse
import datetime
from omegaconf import OmegaConf

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from atari_utils.buffers import CompressedReplayBuffer, CompressedRolloutBuffer


class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        buffer = None
        if hasattr(self.model, "replay_buffer"):
            buffer = self.model.replay_buffer
        elif hasattr(self.model, "rollout_buffer"):
            buffer = self.model.rollout_buffer
        else:
            print("Model has neither a replay_buffer or a rollout_buffer.")
            return False

        # Only log obs_nonzero_count if using compressed buffer
        if hasattr(buffer, "obs_comp"):
            self.logger.record("eval/obs_nonzero_count", buffer.obs_comp.nonzer_count)
        self.logger.record("eval/nbytes", buffer.obs_comp.nbytes)
        return True


def create_tb_log_name(alg, env, seed, compress):
    now = datetime.datetime.now()
    fmt_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{alg}_{env}_{compress}_{seed}_{fmt_now}"
    return log_name


def train(cfg, seed):
    run_name = create_tb_log_name(cfg.alg, cfg.env, seed, cfg.compress)
    log_path = os.path.join(cfg.log, run_name)

    # Create log directory (if it doesn't exist), and save config
    os.makedirs(log_path, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(log_path, "config.yaml"))
    print(f"Running experiment {run_name}")

    n_envs = cfg.PPO.n_envs if cfg.alg == "PPO" else cfg.QRDQN.n_envs

    # Create evaluation callback
    eval_env = make_atari_env(cfg.env, n_envs=n_envs, seed=seed)
    eval_env = VecFrameStack(eval_env, n_stack=cfg.n_stack)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_path,
        log_path=log_path,
        eval_freq=max(cfg.eval.eval_freq // n_envs, 1),
        n_eval_episodes=cfg.eval.n_eval_episodes,
        deterministic=cfg.eval.deterministic,
        verbose=cfg.verbose,
        callback_after_eval=CustomCallback(),
    )

    if cfg.alg == "PPO":
        vec_env = make_atari_env(cfg.env, n_envs=n_envs, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=cfg.n_stack)

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=cfg.verbose,
            n_steps=cfg.PPO.n_steps,
            n_epochs=cfg.PPO.n_epochs,
            batch_size=cfg.PPO.batch_size,
            learning_rate=cfg.PPO.learning_rate,
            clip_range=cfg.PPO.clip_range,
            vf_coef=cfg.PPO.vf_coef,
            ent_coef=cfg.PPO.ent_coef,
            tensorboard_log=cfg.tensorboard_log,
        )
        if cfg.compress:
            model.rollout_buffer = CompressedRolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=model.n_envs,
            )
    elif cfg.alg == "QRDQN":
        vec_env = make_atari_env(cfg.env, n_envs=n_envs, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = QRDQN(
            "CnnPolicy",
            vec_env,
            exploration_fraction=cfg.QRDQN.exploration_fraction,
            buffer_size=cfg.QRDQN.buffer_size,
            optimize_memory_usage=cfg.QRDQN.optimize_memory_usage,
            verbose=cfg.verbose,
            tensorboard_log=cfg.tensorboard_log,
            replay_buffer_class=(CompressedReplayBuffer if cfg.compress else None),
        )
    else:
        print("Invalid alg supplied.")
        return

    model.learn(
        total_timesteps=cfg.total_timesteps,
        tb_log_name=run_name,
        callback=eval_callback,
    )
    model.save(run_name)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="cfgs/experiment.yaml",
        help="Path to config YAML file.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    # Load config parameters
    default_config = OmegaConf.load("cfgs/default.yaml")
    experiment_config = OmegaConf.load("cfgs/experiment.yaml")
    cli_config = OmegaConf.from_cli()
    config = OmegaConf.merge(default_config, experiment_config, cli_config)

    for seed in config.seeds:
        train(config, seed)
