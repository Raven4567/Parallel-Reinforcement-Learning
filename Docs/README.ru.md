# Parallel-Reinforcement-Learning
**Языки**: [English](README.md) | [Русский](README.ru.md) | [Deutsch](README.de.md) | [Español](README.es.md) | [中文](README.zh-CN.md)

## Описание
Небольшая программа, которая заставляет PPO учиться, используя множество сред одновременно, асинхронно, ускоряя обучение и исследование без громоздких процессов multiprocessing, и используя только cuda акселерацию.

## Установка
запустите:
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
затем запустите:
```
pip install -r requirements.txt
```
находясь в установленной папке `.../Parallel-Reinforcement-Learning`.

## Быстрый старт
```python
from PPO import PPO
from AsyncTools.AsyncPPO import AsyncPPO

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

## Параметры `PPO`:

- `is_continuous` - установите True, если среда требует непрерывных действий (False означает дискретные действия, а True означает непрерывные).
- `observ_dim` - количество признаков состояния (например, `observ_dim=4` для CartPole-v1 или `observ_dim=348` для Humanoid-v5).
- `action_dim` - количество возможных действий (например, action_dim=2 для CartPole-v1 или action_dim=23 для Pusher-v5).
- `lr` - значение скорости обучения для оптимизатора.
- `action_scaling` - множитель для действий, например, для Pusher-v5 мы должны использовать - `action_scaling=2.0`, потому что диапазон действий в Pusher-v5 составляет (-2, 2), а наша сеть выводит только действия (-1, 1), если `is_continuous=True`, поэтому она использует `action_scaling` для масштабирования действий в правильный диапазон.
- `policy_clip` - значение изменений политики, например, policy_clip=0.2 допускает изменения не более чем на 20%.
- `k_epochs` - количество эпох для обучения сети на одном наборе данных.
- `GAE_lambda` - коэффициент сглаживания для расчета преимущества (0 = высокая дисперсия, 1 = низкая дисперсия).
- `batch_size` - размер пакета.
- `mini_batch_size` - размер мини-пакета.
- `gamma` - влияет на учет долгосрочных вознаграждений (обычно 0.99-0.999).
- `use_RND` - будем ли мы использовать *Random Network Distillation*.
- `beta` - множитель для вознаграждений `RND`.

Больше об RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Пользовательский цикл
**ВНИМАНИЕ**: строки `env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` и `states = utils.inactive_states_dropout(next_states, dones | truncates)` являются наиболее важными, их отсутствие нарушит обучение.

Также в этом примере я буду использовать свою собственную реализацию, но не стесняйтесь скопировать этот код и переписать его для своей реализации.

```python
# Импорт
from PPO import PPO

from AsyncTools.AsyncPPO import EnvVectorizer, VecMemory
from AsyncTools import utils

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

	env = EnvVectorizer(env=env, num_envs=32) # Векторизованное окружение
	buffer = VecMemory(num_envs=32) # Векторизованный буфер с одним буфером для каждого окружения

	# Цикл сбора данных с индикатором прогресса tqdm
	pbar = tqdm(
		total=100000,
		unit='step'
	)

	while pbar.n < pbar.total:
		states = env.reset()[0] # Получить состояния

		rewards_score = np.array(0.) # Сброс оценки вознаграждений
		steps_score = np.array(0) # Сброс оценки шагов

		while True:
			# Получить действия, значения состояний и логарифмические вероятности
			actions = ppo.get_action(t.from_numpy(states)) 

			# Выполнить шаги
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Добавить данные в буфер AsyncPPO.VecMemory
			utils.buffer_append(
				buffer,

				states, 
				actions, 
				rewards, 
				dones | truncates,

				is_env_terminal=env.envs_active,
				num_envs=32
			) 

			# Отбор состояний с признаками done или truncate = True, а также обновление списка активных окружений
			states = utils.inactive_states_dropout(next_states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Обновить оценку вознаграждений
			steps_score += sum(~env.envs_active) # Обновить оценку шагов

			# Если все окружения терминальны, мы завершаем эпизод
			if np.all(env.envs_active): 
				# Передача данных из нашего локального буфера в буфер ppo.memory для обучения ppo. Вы также можете использовать свою собственную функцию для передачи данных в буфер вашей собственной нейронной сети.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				ppo.learn() # Запустить функцию обучения

				pbar.set_description(f'Mean reward {rewards_score / 32: .1f}')
				pbar.update(min(pbar.total - pbar.n, steps_score))

				break # Выйти из эпизода и начать новый
```