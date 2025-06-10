# Parallel-Reinforcement-Learning

**Idiomas:** Inglés | Ruso | Alemán | Español | Chino

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
estando en la carpeta `Parallel-Reinforcement-Learning` instalada.

## Inicio rápido
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

	async_ppo.ppo.save_weights(path='(inserta tu ruta)/PPO_PRL/PPO/data/')
```

Parámetros de `PPO`:

- `is_continuous` - establece True si el entorno requiere acciones continuas (False significa acciones discretas y True significa continuas).
- `action_dim`  - número de acciones posibles (p. ej., `action_dim=2` para CartPole-v1 o `action_dim=23` para Pusher-v5).
- `observ_dim` - número de características del estado (p. ej., `observ_dim=4` para CartPole-v1 o `observ_dim=348` para Humanoid-v5).
- `Actor_lr` - valor de lr para la red Actor.
- `Critic_lr` - valor de lr para la red Critic.
- `action_scaling` - multiplicador para las acciones, por ejemplo, para Pusher-v5 tenemos que usar `action_scaling=2.0` porque el rango de acciones en Pusher-v5 es (-2, 2) y nuestra red solo emite acciones (-1, 1) si `is_continuous=True`, por lo que usa `action_scaling` para escalar las acciones al rango correcto.
- `policy_clip` - valor de los cambios de política, p. ej., `policy_clip=0.2` permite cambios no superiores al 20%.
- `k_epochs` - número de épocas para el aprendizaje de la red sobre un conjunto de datos.
- `GAE_lambda` - factor de suavizado para el cálculo de la ventaja (0 = alta varianza, 1 = menor varianza).
- `batch_size` - tamaño del lote.
- `mini_batch_size` - tamaño del mini-lote.
- `gamma` - afecta la consideración de recompensas a largo plazo (generalmente 0.99-0.999).
- `use_RND` - si usaremos `Random Network Distillation`.
- `beta` - multiplicador para las recompensas de `RND`.

Más sobre RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Bucle personalizado

**ADVERTENCIA**: la línea `env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` es muy importante, su ausencia interrumpirá el aprendizaje.

También en este ejemplo usaré mi propia implementación, pero siéntete libre de copiar este código y reescribirlo para tu implementación.

```python
# Importar
from PPO import PPO
from AsyncPPO import EnvVectorizer, VecMemory

import utils

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
		is_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		Actor_lr=0.0010, Critic_lr=0.0025,# action_scaling=1.0,
		policy_clip=0.2, k_epochs=11, GAE_lambda=0.95, 
		batch_size=1024, mini_batch_size=1024, gamma=0.995,
		use_RND=True, beta=0.01
	)

	env = EnvVectorizer(env=env, num_envs=32) # Entorno vectorizado
	buffer = VecMemory(num_envs=32) # Búfer vectorizado con un búfer para cada entorno

	# Bucle de recopilación de datos con barra de progreso tqdm
	for _ in (pbar := tqdm(range(200))):
		states = env.reset()[0] # Obtener estados

		rewards_score = np.array(0.) # Restablecer puntuación de recompensas
		steps_score = np.array(0) # Restablecer puntuación de pasos

		while True:
			# Obtener acciones, valores de estado y logaritmos de probabilidades
			actions, state_values, log_probs = ppo.get_action(t.from_numpy(states)) 

			# Ejecutar pasos
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Añadir datos a nuestro búfer AsyncPPO.VecMemory
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

			# Filtrar estados con características done o truncate = True, y también actualizar la lista de actividad de los entornos
			states = utils.inactive_states_dropout(states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Actualizar puntuación de recompensas
			steps_score += len(actions) # Actualizar puntuación de pasos

			# Si todos los entornos son terminales, finalizamos el episodio
			if np.all(env.envs_active): 
				# Transferir datos de nuestro búfer local al búfer ppo.memory para el aprendizaje de PPO. También puedes usar tu propia función para transferir datos al búfer de tu propia red neuronal.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				
				ppo.learn() # Iniciar la función de aprendizaje

				break # Salir del episodio y comenzar uno nuevo
```