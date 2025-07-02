# Parallel-Reinforcement-Learning

**语言**: [英语](README.md) | [俄语](README.ru.md) | [德语](README.de.md) | [西班牙语](README.es.md) | [中文](README.zh-CN.md)

## 描述
一个小程序，使 PPO 可以同时异步地使用多个环境进行学习，从而加速学习和探索过程。

## 安装
运行：
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
然后运行：
```
pip install -r requirements.txt
```
在已安装的 Parallel-Reinforcement-Learning 文件夹中。

## 快速开始
```python
from PPO import PPO
from AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	env = gym.make('CartPole-v1', max_episode_steps=500)

	ppo = PPO(
		is_continuous=False, 
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n, 
		lr=0.001, 
		# action_scaling=2.0
		policy_clip=0.2, 
		k_epochs=11, 
		GAE_lambda=0.95, 
		batch_size=1024, 
		mini_batch_size=512, 
		gamma=0.995,
		# use_RND=True, 
		# beta=0.001
	)

	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=100000
	)

	async_ppo.run()

	async_ppo.ppo.save_weights(path='(insert your path)/Parallel-Reinforcement-Learning/PPO/data')
```

## ```PPO` 参数:

- `is_continuous` - 如果环境需要连续动作，则设置为 True (False 表示离散动作，True 表示连续动作)。
- `observ_dim` - 状态特征的数量 (例如，CartPole-v1 为 `observ_dim=4`，Humanoid-v5 为 `observ_dim=348`)。
- `action_dim`  - 可能的动作数量 (例如，CartPole-v1 为 `action_dim=2`，Pusher-v5 为 `action_dim=23`)。
- `lr` - 优化器的学习率 (lr) 值。
- `action_scaling` - 动作的乘数，例如对于 Pusher-v5，我们必须使用 `action_scaling=2.0`，因为 Pusher-v5 中的动作范围是 (-2, 2)，而如果 `is_continuous=True`，我们的网络仅输出 (-1, 1) 的动作，因此它使用 `action_scaling` 将动作缩放到正确的范围。
- `policy_clip` - 策略更改的值，例如 `policy_clip=0.2` 允许不超过 20% 的更改。
- `k_epochs` - 网络在一组数据上学习的时期数。
- `GAE_lambda` -优势计算的平滑因子 (0 = 高方差, 1 = 低方差)。
- `batch_size` - 批量大小。
- `mini_batch_size` - 小批量大小。
- `gamma` - 影响长期奖励的考虑 (通常为 0.99-0.999)。
- `use_RND` - 我们是否将使用 `Random Network Distillation` (随机网络蒸馏)。
- `beta` - `RND` 奖励的乘数。

关于 RND 的更多信息 - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## 自定义循环
**警告**：`env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` 和 `states = utils.inactive_states_dropout(next_states, dones | truncates)` 这两行最为重要，缺少它们会破坏学习过程。

此外，在此示例中，我将使用我自己的实现，但您可以随意复制此代码并为您的实现重写它。

```python
# 导入
from PPO import PPO
from AsyncPPO import EnvVectorizer, VecMemory

import utils

import torch as t
import numpy as np

from tqdm import tqdm

import gymnasium as gym

# 主循环
if __name__ == '__main__':
	# 创建环境
	env = gym.make('CartPole-v1')

	# 初始化神经网络 (或您自己的实现)
	ppo = PPO(
		is_continuous=False, 
		observ_dim=env.observation_space.shape[0],
		action_dim=env.action_space.n, 
		lr=0.001,
		# action_scaling=1.0,
		policy_clip=0.2, 
		k_epochs=11, 
		GAE_lambda=0.95, 
		batch_size=1024, 
		mini_batch_size=512, 
		gamma=0.995,
		# use_RND=True, 
		# beta=0.001
	)

	env = EnvVectorizer(env=env, num_envs=32) # 向量化环境
	buffer = VecMemory(num_envs=32) # 向量化缓冲区，每个环境一个缓冲区

	# 带 tqdm 进度条的数据收集循环
	pbar = tqdm(
		total=100000,
		unit='step'
	)

	while pbar.n < pbar.total:
		states = env.reset()[0] # 获取状态

		rewards_score = np.array(0.) # 重置奖励分数
		steps_score = np.array(0) # 重置步数分数

		while True:
			# 获取动作、状态值和对数概率
			actions = ppo.get_action(t.from_numpy(states)) 

			# 执行步骤
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# 将数据附加到我们的 AsyncPPO.VecMemory 缓冲区中
			utils.buffer_append(
				buffer,

				states, 
				actions, 
				rewards, 
				dones | truncates,

				is_env_terminal=env.envs_active,
				num_envs=32
			) 

			# 筛选具有 done 或 truncate = True 特征的状态，并更新环境活动列表
			states = utils.inactive_states_dropout(next_states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # 更新奖励分数
			steps_score += sum(~env.envs_active) # 更新步数分数

			# 如果所有环境都已终止，我们结束该回合
			if np.all(env.envs_active): 
				# 将数据从我们的本地缓冲区传输到 ppo.memory 缓冲区以进行 PPO 学习。您也可以使用自己的函数将数据传输到您自己的神经网络缓冲区中。
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				ppo.learn() # 启动学习函数

				pbar.set_description(f'Mean reward {rewards_score / 32: .1f}')
				pbar.update(min(pbar.total - pbar.n, steps_score))

				break # 退出当前回合并开始新的回合
