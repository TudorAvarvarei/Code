from stable_baselines3 import PPO, A2C, SAC, TD3
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.results_plotter import load_results, ts2xy
from gymnasium.envs.classic_control.pendulum import PendulumEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import numpy as np
import time
import matplotlib as plt
from stable_baselines3.common.callbacks import BaseCallback
import os
from tqdm.auto import tqdm
from gymnasium import spaces


### Create functions within stable baselines---------------------------------------------------------------------------------------

# class TimeLimitWrapper(gym.Wrapper):
#     """
#     :param env: (gym.Env) Gym environment that will be wrapped
#     :param max_steps: (int) Max number of steps per episode
#     """

#     def __init__(self, env, max_steps=100):
#         # Call the parent constructor, so we can access self.env later
#         super(TimeLimitWrapper, self).__init__(env)
#         self.max_steps = max_steps
#         # Counter of steps per episode
#         self.current_step = 0

#     def reset(self, **kwargs):
#         """
#         Reset the environment
#         """
#         # Reset the counter
#         self.current_step = 0
#         return self.env.reset(**kwargs)

#     def step(self, action):
#         """
#         :param action: ([float] or int) Action taken by the agent
#         :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode over?, additional informations
#         """
#         self.current_step += 1
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         # Overwrite the truncation signal when when the number of steps reaches the maximum
#         if self.current_step >= self.max_steps:
#             truncated = True
#         return obs, reward, terminated, truncated, info

### Create basic stuff like environments, models and how to save / load them------------------------------------------------------------------------

# env = gym.make("CartPole-v1")

# model = PPO('MlpPolicy', "CartPole-v1", verbose=1).learn(10)

# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100, warn=False)

# print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# model.save(f"models/PPO_tutorial")

# Here we create the environment directly because gym.make() already wrap the environment in a TimeLimit wrapper otherwise
# env = PendulumEnv()
# # Wrap the environment
# env = TimeLimitWrapper(env, max_steps=100)

# obs, _ = env.reset()
# done = False
# n_steps = 0
# while not done:
#     # Take random actions
#     random_action = env.action_space.sample()
#     obs, reward, terminated, truncated, info = env.step(random_action)
#     done = terminated or truncated
#     n_steps += 1

# print(n_steps, info)

### Multiprocessing----------------------------------------------------------------------------------------------------

# env_id = "CartPole-v1"
# # The different number of processes that will be used
# PROCESSES_TO_TEST = [1, 2, 4, 8, 16]
# NUM_EXPERIMENTS = 3  # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
# TRAIN_STEPS = 5000
# # Number of episodes for evaluation
# EVAL_EPS = 20
# ALGO = A2C

# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """

#     def _init():
#         env = gym.make(env_id)
#         # use a seed for reproducibility
#         # Important: use a different seed for each environment
#         # otherwise they would generate the same experiences
#         env.reset(seed=seed + rank)
#         return env

#     set_random_seed(seed)
#     return _init

# # We will create one environment to evaluate the agent on
# eval_env = gym.make(env_id)

# reward_averages = []
# reward_std = []
# training_times = []
# total_procs = 0

# for n_procs in PROCESSES_TO_TEST:
#     total_procs += n_procs
#     print(f"Running for n_procs = {n_procs}")
#     if n_procs == 1:
#         # if there is only one process, there is no need to use multiprocessing
#         train_env = DummyVecEnv([lambda: gym.make(env_id)])
#     else:
#         # Here we use the "fork" method for launching the processes, more information is available in the doc
#         # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
#         train_env = SubprocVecEnv(
#             [make_env(env_id, i + total_procs) for i in range(n_procs)], start_method="fork",
#         )

#     rewards = []
#     times = []

#     for experiment in range(NUM_EXPERIMENTS):
#         # it is recommended to run several experiments due to variability in results
#         train_env.reset()
#         model = ALGO("MlpPolicy", train_env, verbose=0)
#         start = time.time()
#         model.learn(total_timesteps=TRAIN_STEPS)
#         times.append(time.time() - start)
#         mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
#         rewards.append(mean_reward)
#     # Important: when using subprocesses, don't forget to close them
#     # otherwise, you may have memory issues when running a lot of experiments
#     train_env.close()
#     reward_averages.append(np.mean(rewards))
#     reward_std.append(np.std(rewards))
#     training_times.append(np.mean(times))

# def plot_training_results(training_steps_per_second, reward_averages, reward_std):
#     """
#     Utility function for plotting the results of training

#     :param training_steps_per_second: List[double]
#     :param reward_averages: List[double]
#     :param reward_std: List[double]
#     """
#     plt.figure(figsize=(9, 4))
#     plt.subplots_adjust(wspace=0.5)
#     plt.subplot(1, 2, 1)
#     plt.errorbar(
#         PROCESSES_TO_TEST,
#         reward_averages,
#         yerr=reward_std,
#         capsize=2,
#         c="k",
#         marker="o",
#     )
#     plt.xlabel("Processes")
#     plt.ylabel("Average return")
#     plt.subplot(1, 2, 2)
#     plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
#     plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
#     plt.xlabel("Processes")
#     plt.ylabel("Training steps per second")

# training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

# plot_training_results(training_steps_per_second, reward_averages, reward_std)

### Hyperparamter optimisation----------------------------------------------------------------------------------------------------------
# automatic hyperparameter optimization using Optuna

# eval_env = gym.make("Pendulum-v1")

# default_model = SAC(
#     "MlpPolicy",
#     "Pendulum-v1",
#     verbose=1,
#     seed=0,
#     batch_size=64,
#     policy_kwargs=dict(net_arch=[64, 64]),
# ).learn(8000)

# mean_reward, std_reward = evaluate_policy(default_model, eval_env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

# tuned_model = SAC(
#     "MlpPolicy",
#     "Pendulum-v1",
#     batch_size=256,
#     verbose=1,
#     policy_kwargs=dict(net_arch=[256, 256]),
#     seed=0,
# ).learn(8000)

# mean_reward, std_reward = evaluate_policy(tuned_model, eval_env, n_eval_episodes=100)
# print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


### Callbacks-------------------------------------------------------------------------------------------------------------------

# class SimpleCallback(BaseCallback):
#     """
#     a simple callback that can only be called twice

#     :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
#     """

#     def __init__(self, verbose=0):
#         super(SimpleCallback, self).__init__(verbose)
#         self._called = False

#     def _on_step(self):
#         if not self._called:
#             print("callback - first call")
#             self._called = True
#             return True  # returns True, training continues.
#         print("callback - second call")
#         return False  # returns False, training stops.
    
# model = SAC("MlpPolicy", "Pendulum-v1", verbose=1)
# model.learn(8000, callback=SimpleCallback())


# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq: (int)
#     :param log_dir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """

#     def __init__(self, check_freq, log_dir, verbose=1):
#         super().__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, "best_model")
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq == 0:

#             # Retrieve training reward
#             x, y = ts2xy(load_results(self.log_dir), "timesteps")
#             if len(x) > 0:
#                 # Mean training reward over the last 100 episodes
#                 mean_reward = np.mean(y[-100:])
#                 if self.verbose > 0:
#                     print("Num timesteps: {}".format(self.num_timesteps))
#                     print(
#                         "Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(
#                             self.best_mean_reward, mean_reward
#                         )
#                     )

#                 # New best model, you could save the agent here
#                 if mean_reward > self.best_mean_reward:
#                     self.best_mean_reward = mean_reward
#                     # Example for saving best model
#                     if self.verbose > 0:
#                         print("Saving new best model at {} timesteps".format(x[-1]))
#                         print("Saving new best model to {}.zip".format(self.save_path))
#                     self.model.save(self.save_path)

#         return True
    
# # Create log dir
# log_dir = "/models"
# os.makedirs(log_dir, exist_ok=True)

# # Create and wrap the environment
# env = make_vec_env("CartPole-v1", n_envs=1, monitor_dir=log_dir)

# # Create Callback
# callback = SaveOnBestTrainingRewardCallback(check_freq=20, log_dir=log_dir, verbose=1)

# model = A2C("MlpPolicy", env, verbose=0)
# model.learn(total_timesteps=5000, callback=callback)

# class ProgressBarCallback(BaseCallback):
#     """
#     :param pbar: (tqdm.pbar) Progress bar object
#     """

#     def __init__(self, pbar):
#         super().__init__()
#         self._pbar = pbar

#     def _on_step(self):
#         # Update the progress bar:
#         self._pbar.n = self.num_timesteps
#         self._pbar.update(0)


# # this callback uses the 'with' block, allowing for correct initialisation and destruction
# class ProgressBarManager(object):
#     def __init__(self, total_timesteps):  # init object with total timesteps
#         self.pbar = None
#         self.total_timesteps = total_timesteps

#     def __enter__(self):  # create the progress bar and callback, return the callback
#         self.pbar = tqdm(total=self.total_timesteps)

#         return ProgressBarCallback(self.pbar)

#     def __exit__(self, exc_type, exc_val, exc_tb):  # close the callback
#         self.pbar.n = self.total_timesteps
#         self.pbar.update(0)
#         self.pbar.close()


# model = TD3("MlpPolicy", "Pendulum-v1", verbose=0)
# # Using a context manager garanties that the tqdm progress bar closes correctly
# with ProgressBarManager(2000) as callback:
#     model.learn(2000, callback=callback)

### Create your own environment--------------------------------------------------------------------------------------------

class GoLeftEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """

    # Because of google colab, we cannot implement the GUI ('human' render mode)
    metadata = {"render_modes": ["console"]}

    # Define constants for clearer code
    LEFT = 0
    RIGHT = 1

    def __init__(self, grid_size=10, render_mode="console"):
        super(GoLeftEnv, self).__init__()
        self.render_mode = render_mode

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = grid_size - 1

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Discrete(n_actions)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size, shape=(1,), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        super().reset(seed=seed, options=options)
        # Initialize the agent at the right of the grid
        self.agent_pos = self.grid_size - 1
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return np.array([self.agent_pos]).astype(np.float32), {}  # empty info dict

    def step(self, action):
        if action == self.LEFT:
            self.agent_pos -= 1
        elif action == self.RIGHT:
            self.agent_pos += 1
        else:
            raise ValueError(
                f"Received invalid action={action} which is not part of the action space"
            )

        # Account for the boundaries of the grid
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size)

        # Are we at the left of the grid?
        terminated = bool(self.agent_pos == 0)
        truncated = False  # we do not limit the number of steps here

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = 1 if self.agent_pos == 0 else 0

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        return (
            np.array([self.agent_pos]).astype(np.float32),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        # agent is represented as a cross, rest as a dot
        if self.render_mode == "console":
            print("." * self.agent_pos, end="")
            print("x", end="")
            print("." * (self.grid_size - self.agent_pos))

    def close(self):
        pass

env = GoLeftEnv(grid_size=10)
vec_env = make_vec_env(GoLeftEnv, n_envs=1, env_kwargs=dict(grid_size=10))

model = A2C("MlpPolicy", env, verbose=1).learn(5000)


# Test the trained agent
# using the vecenv
obs = vec_env.reset()
n_steps = 20
for step in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    print(f"Step {step + 1}")
    print("Action: ", action)
    obs, reward, done, info = vec_env.step(action)
    print("obs=", obs, "reward=", reward, "done=", done)
    vec_env.render()
    if done:
        # Note that the VecEnv resets automatically
        # when a done signal is encountered
        print("Goal reached!", "reward=", reward)
        break