import argparse
import datetime

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import PPO
from sb3_contrib import QRDQN
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from atari_utils.buffers import CompressedReplayBuffer, CompressedRolloutBuffer


ALGS = ["PPO", "QRDQN"]
ENVS = [
    "AsteroidsNoFrameskip-v4",
    "BeamRiderNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
]
SEEDS = [1, 2, 3, 4, 5]
COMPRESS = [False, True]


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

        # Only log obs_nonzero_count if using quantized/compressed buffer
        if hasattr(buffer, "obs_comp"):
            self.logger.record(
                "eval/obs_nonzero_count", buffer.obs_comp.get_nonzero_count()
            )
        return True


def create_tb_log_name(alg, env, seed, compress):
    now = datetime.datetime.now()
    fmt_now = now.strftime("%Y-%m-%d_%H-%M-%S")
    log_name = f"{alg}_{env}_{compress}_{seed}_{fmt_now}"
    return log_name


def train(alg, env, seed, compress, total_timesteps=1e7):
    run_name = create_tb_log_name(alg, env, seed, compress)
    print(f"Running experiment {run_name}")

    n_envs = 8 if alg == "PPO" else 1

    # Create evaluation callback
    eval_env = make_atari_env(env, n_envs=1, seed=seed)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"./logs/{run_name}",
        log_path=f"./logs/{run_name}",
        eval_freq=max(50000 // n_envs, 1),
        n_eval_episodes=3,
        deterministic=True,
        verbose=0,
        callback_after_eval=CustomCallback(),
    )

    if alg == "PPO":
        vec_env = make_atari_env(env, n_envs=n_envs, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=0,
            n_steps=128,
            n_epochs=4,
            batch_size=256,
            learning_rate=2.5e-4,
            clip_range=0.1,
            vf_coef=0.5,
            ent_coef=0.01,
            tensorboard_log="./tensorboard_logs/",
        )
        if compress:
            model.rollout_buffer = CompressedRolloutBuffer(
                model.n_steps,
                model.observation_space,
                model.action_space,
                device=model.device,
                gamma=model.gamma,
                gae_lambda=model.gae_lambda,
                n_envs=model.n_envs,
            )
    elif alg == "QRDQN":
        vec_env = make_atari_env(env, n_envs=n_envs, seed=seed)
        vec_env = VecFrameStack(vec_env, n_stack=4)

        model = QRDQN(
            "CnnPolicy",
            vec_env,
            exploration_fraction=0.025,
            buffer_size=100000,
            optimize_memory_usage=False,
            verbose=0,
            tensorboard_log="./tensorboard_logs/",
            replay_buffer_class=(CompressedReplayBuffer if compress else None),
        )
    else:
        print("Invalid alg supplied.")
        return

    model.learn(total_timesteps=total_timesteps, tb_log_name=run_name, callback=eval_callback)
    model.save(run_name)


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alg", type=str, default="PPO", help="Name of algorithm to use (PPO, QRDQN)."
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Atari gym environment to run.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[1], help="List of random seeds."
    )
    parser.add_argument(
        "--compress", action="store_true", help="Use observation space compression."
    )
    parser.add_argument("--all", action="store_true", help="Run all experiments for 500k steps.")

    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    if not args.all:
        if not args.env:
            raise ValueError("No env supplied!")
        for seed in args.seeds:
            train(args.alg, args.env, seed, args.compress)
    elif args.all and args.env is not None:
        assert args.env in ENVS
        print(f"Running all experiments for env {args.env}")
        for alg in ALGS:
            for compress in COMPRESS:
                for seed in SEEDS:
                    train(alg, args.env, seed, compress)
    else:
        print("Running all experiments!")
        for alg in ALGS:
            for env in ENVS:
                for compress in COMPRESS:
                    for seed in SEEDS:
                        train(alg, env, seed, compress)
