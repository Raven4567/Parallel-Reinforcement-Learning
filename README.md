# Parallel-Reinforcement-Learning

## The PPO parameters
```python
has_continuous: bool  # Whether the gym environment is discrete or continuous

action_dim: int  # Number of possible actions

observ_dim: int  # Number of environment features

action_scaling: float = None  # For example, if the range of actions is from -2 to 2, but our network outputs actions in the range of -1 to 1, we multiply the actions by action_scaling.

actor_lr: float = 0.001  # Actor's optimizer learning rate

critic_lr: float = 0.0025  # Critic's optimizer learning rate

k_epochs: int = 23  # Number of epochs for updating PPO policies

policy_clip: float = 0.2  # The limit on policy updates

GAE_lambda: float = 0.95

gamma: float = 0.995  # This controls the focus on long-term rewards. If it's lower, the agent focuses on short-term rewards, and if it's higher, the agent focuses on long-term rewards. Typically between 0.9 and 0.999.

batch_size: int = 1024  # Size of the batch

mini_batch_size: int = 512  # Size of the mini-batch

use_RND: bool = False  # Whether to use RND (Random Network Distillation)

beta: int = None  # Impact of RND's rewards

num_workers: int = 1  # Number of parallel environments. Choose between 4 and 32 agents.

# Training parameters
EPISODES: int  # Number of episodes for training

NUM_WORKERS: int  # Number of workers for parallel learning. This is also passed to PPO.__init__
```
**Read more: [The explanation of RND](https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/#main)**

## Why Threading?
In this code, I use Python's `threading` because it is **cheap** and *doesn’t introduce the overhead that processes do, such as the communication and synchronization of network weights between processes.* Processes work at the same speed and **don’t face the data race problem**. However, `PyTorch` is thread-safe, **so even with dozens of environments, it doesn’t cause data race problems.** As mentioned, `threading` avoids the overhead because it *uses one interpreter*. Using a single interpreter may slow down execution, but we prefer `threading` over `multiprocessing` because it's lighter than independent processes, with threads sharing **one memory** and **one model.**

### Code Components
The code uses **PPO** as the reinforcement learning algorithm. It employs Python's `threading` to realize asynchronous training of the `PPO` agent.

### Working Principle
We initialize the `ppo` and `graphic` variables, with `ppo` used for learning and predicting actions. We also initialize `EPISODES` and `NUM_WORKERS`. Then, we run a for-loop for the number of `EPISODES`. We create `NUM_WORKERS` threads, with each thread corresponding to one environment. After each Thread completes, we re-draw the graph and call `ppo.education()`, which updates the policy. The loop repeats.

### Principle of ppo.education's Work
If `ppo.num_workers > 1`, we call `ppo.transfer_of_data` to move data from `ppo.sub_buffer` to `ppo.memory`. Then, we convert the `ppo.memory` lists into `torch.tensor` objects on the `device`, which could be `cuda` or `cpu`.

Next, we initialize rewards with or without `RND`'s intrinsic rewards, and set up the `dones` array. We then compute **`GAE rewards`** and execute the usual `k_epoch` cycle for updating the policy.

#### Architecture
In `PPO`, we use **a simple perceptron architecture** with **two hidden layers**, **each having 64 neurons**. This is **the most average** architecture for OpenAI gym environments. Additionally, we place the `torch.nn.ReLU6` activation function between the layers.