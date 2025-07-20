# Parallel-Reinforcement-Learning

**Idiomas**: [Inglés](README.md) | [Ruso](README.ru.md) | [Alemán](README.de.md) | [Español](README.es.md) | [Chino](README.zh-CN.md)

## Descripción
Pequeño programa que hace que PPO aprenda usando muchos entornos al mismo tiempo, de forma asíncrona, acelerando el aprendizaje y la exploración.

## Instalación
ejecuta:
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
luego ejecuta:
```
pip install -r requirements.txt
```
estando en la carpeta Parallel-Reinforcement-Learning instalada.

## Inicio rápido
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

## Parámetros de PPO:

- `is_continuous` - establece True si el entorno requiere acciones continuas (False significa acciones discretas y True significa continuas).
- `observ_dim` - número de características del estado (p. ej., `observ_dim=4` para CartPole-v1 o `observ_dim=348` para Humanoid-v5).
- `action_dim`  - número de acciones posibles (p. ej., `action_dim=2` para CartPole-v1 o `action_dim=23` para Pusher-v5).
`lr` - valor de la tasa de aprendizaje para el optimizador.
- `action_scaling` - multiplicador para las acciones, por ejemplo, para Pusher-v5 tenemos que usar `action_scaling=2.0` porque el rango de acciones en Pusher-v5 es (-2, 2) y nuestra red solo emite acciones (-1, 1) si - `is_continuous=True`, por lo que usa `action_scaling` para escalar las acciones al rango correcto.
- `policy_clip` - valor de los cambios de política, p. ej., policy_clip=0.2 permite cambios no superiores al 20%.
- `k_epochs` - número de épocas para el aprendizaje de la red sobre un conjunto de datos.
- `GAE_lambda` - factor de suavizado para el cálculo de la ventaja (0 = alta varianza, 1 = menor varianza).
- `batch_size` - tamaño del lote.
- `mini_batch_size` - tamaño del mini-lote.
- `gamma` - afecta la consideración de recompensas a largo plazo (generalmente 0.99-0.999).
- `use_RND` - si usaremos *Random Network Distillation*.
- `beta` - multiplicador para las recompensas de `RND`.

Más sobre RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Bucle personalizado
**ADVERTENCIA**: las líneas `env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` y `states = utils.inactive_states_dropout(next_states, dones | truncates)` son muy importantes, su ausencia interrumpirá el aprendizaje.

También en este ejemplo usaré mi propia implementación, pero siéntete libre de copiar este código y reescribirlo para tu implementación.

```python
# Importar
from PPO import PPO

from AsyncTools.AsyncPPO import EnvVectorizer, VecMemory
from AsyncTools import utils

import torch as t
import numpy as np

from tqdm import tqdm

import gymnasium as gym

# Bucle principal
if __name__ == '__main__':
	# Crear entorno
	env = gym.make('CartPole-v1')

	# Inicializar red neuronal (o tu propia implementación)
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

	env = EnvVectorizer(env=env, num_envs=32) # Entorno vectorizado
	buffer = VecMemory(num_envs=32) # Búfer vectorizado con un búfer para cada entorno

	# Bucle de recopilación de datos con barra de progreso tqdm
	pbar = tqdm(
		total=100000,
		unit='step'
	)

	while pbar.n < pbar.total:
		states = env.reset()[0] # Obtener estados

		rewards_score = np.array(0.) # Restablecer puntuación de recompensas
		steps_score = np.array(0) # Restablecer puntuación de pasos

		while True:
			# Obtener acciones, valores de estado y logaritmos de probabilidades
			actions = ppo.get_action(t.from_numpy(states)) 

			# Ejecutar pasos
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Añadir datos a nuestro búfer AsyncPPO.VecMemory
			utils.buffer_append(
				buffer,

				states, 
				actions, 
				rewards, 
				dones | truncates,

				is_env_terminal=env.envs_active,
				num_envs=32
			) 

			# Filtrar estados con características done o truncate = True, y también actualizar la lista de actividad de los entornos
			states = utils.inactive_states_dropout(next_states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Actualizar puntuación de recompensas
			steps_score += sum(~env.envs_active) # Actualizar puntuación de pasos

			# Si todos los entornos son terminales, finalizamos el episodio
			if np.all(env.envs_active): 
				# Transferir datos de nuestro búfer local al búfer ppo.memory para el aprendizaje de PPO. También puedes usar tu propia función para transferir datos al búfer de tu propia red neuronal.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				ppo.learn() # Iniciar la función de aprendizaje

				pbar.set_description(f'Mean reward {rewards_score / 32: .1f}')
				pbar.update(min(pbar.total - pbar.n, steps_score))

				break # Salir del episodio y comenzar uno nuevo
```
