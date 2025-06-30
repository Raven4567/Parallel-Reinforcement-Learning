# Parallel-Reinforcement-Learning
**Языки**: [English](README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Español](README.es.md) | [中文](README.zh-CN.md)

## Описание
Небольшая программа, которая заставляет PPO обучаться, используя множество окружений одновременно, асинхронно, ускоряя обучение и исследование.

## Установка
запустите:
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
затем запустите:
```
pip install -r requirements.txt
```
находясь в установленной папке `Parallel-Reinforcement-Learning`.

## Быстрый старт
```python
from PPO import PPO
from AsyncPPO import AsyncPPO

import gymnasium as gym

if __name__ == '__main__':
	env = gym.make('CartPole-v1', max_episode_steps=500)

	ppo = PPO(
		is_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		Actor_lr=0.0010, Critic_lr=0.0025, # action_scaling=2.0
		policy_clip=0.2, k_epochs=11, GAE_lambda=0.95, 
		batch_size=1024, mini_batch_size=1024, gamma=0.995,
		# use_RND=True, beta=0.01
	)

	async_ppo = AsyncPPO(
		env=env,
		ppo=ppo,
		num_envs=32,
		steps=1000000
	)

	async_ppo.run()

	async_ppo.ppo.save_weights(path='(insert your path)/PPO_PRL/PPO/data/')
```

## Параметры PPO:

- `is_continuous` - установите True, если среда требует непрерывных действий (False означает дискретные действия, а True означает непрерывные).
- `action_dim` - количество возможных действий (например, action_dim=2 для CartPole-v1 или action_dim=23 для Pusher-v5).
- `observ_dim` - количество признаков состояния (например, observ_dim=4 для CartPole-v1 или observ_dim=348 для Humanoid-v5).
- `actor_lr` - значение lr для сети Actor.
- `critic_lr`  - значение lr для сети Critic.
- `action_scaling` - множитель для действий, например, для Pusher-v5 мы должны использовать action_scaling=2.0, потому что диапазон действий в Pusher-v5 составляет (-2, 2), а наша сеть выводит только действия (-1, 1), если - `is_continuous`=True, поэтому она использует action_scaling для масштабирования действий в правильный диапазон.
- `policy_clip` - значение изменений политики, например, policy_clip=0.2 допускает изменения не более чем на 20%.
- `k_epochs` - количество эпох для обучения сети на одном наборе данных.
- `GAE_lambda` - коэффициент сглаживания для расчета преимущества (0 = высокая дисперсия, 1 = низкая дисперсия).
- `batch_size` - размер пакета.
- `mini_batch_size` - размер мини-пакета.
- `gamma` - влияет на учет долгосрочных вознаграждений (обычно 0.99-0.999).
- `use_RND` - будем ли мы использовать Random Network Distillation.
- `beta` - множитель для вознаграждений RND.

Больше о RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Пользовательский цикл
**ВНИМАНИЕ**: строка env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates) является наиболее важной, ее отсутствие нарушит обучение.

Также в этом примере я буду использовать свою собственную реализацию, но не стесняйтесь скопировать этот код и переписать его для своей реализации.

```python
# Импорт
from PPO import PPO
from AsyncPPO import EnvVectorizer, VecMemory

import utils

import torch as t
import numpy as np

from tqdm import tqdm

import gymnasium as gym

# Основной цикл
if __name__ == '__main__':
	# Создание окружения
	env = gym.make('CartPole-v1')

	# Инициализация нейронной сети (или вашей собственной реализации)
	ppo = PPO(
		is_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		Actor_lr=0.0010, Critic_lr=0.0025,# action_scaling=1.0,
		policy_clip=0.2, k_epochs=11, GAE_lambda=0.95, 
		batch_size=1024, mini_batch_size=1024, gamma=0.995,
		use_RND=True, beta=0.01
	)

	env = EnvVectorizer(env=env, num_envs=32) # Векторизованное окружение
	buffer = VecMemory(num_envs=32) # Векторизованный буфер с одним буфером для каждого окружения

	# Цикл сбора данных с индикатором прогресса tqdm
	for _ in (pbar := tqdm(range(200))):
		states = env.reset()[0] # Получить состояния

		rewards_score = np.array(0.) # Сброс оценки вознаграждений
		steps_score = np.array(0) # Сброс оценки шагов

		while True:
			# Получить действия, значения состояний и логарифмические вероятности
			actions, state_values, log_probs = ppo.get_action(t.from_numpy(states)) 

			# Выполнить шаги
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Добавить данные в буфер AsyncPPO.VecMemory
			utils.buffer_append(
				buffer,

				states, 
				actions, 
				rewards, 
				dones, 
				state_values, 
				log_probs,

				is_env_terminal=env.envs_active,
				num_envs=32
			) 

			# Отбор состояний с признаками done или truncate = True, а также обновление списка активных окружений
			states = utils.inactive_states_dropout(states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Обновить оценку вознаграждений
			steps_score += len(actions) # Обновить оценку шагов

			# Если все окружения терминальны, мы завершаем эпизод
			if np.all(env.envs_active): 
				# Передача данных из нашего локального буфера в буфер ppo.memory для обучения ppo. Вы также можете использовать свою собственную функцию для передачи данных в буфер вашей собственной нейронной сети.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				
				ppo.learn() # Запустить функцию обучения

				break # Выйти из эпизода и начать новый
```