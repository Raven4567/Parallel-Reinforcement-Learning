# Parallel-Reinforcement-Learning

**Sprachen:** Englisch | Russisch | Deutsch | Spanisch | Chinesisch

## Beschreibung
Kleines Programm, das PPO dazu bringt, unter Verwendung vieler Umgebungen gleichzeitig und asynchron zu lernen, wodurch das Lernen und die Exploration beschleunigt werden.

## Installation
führe aus:
```
git clone https://github.com/Raven4567/Parallel-Reinforcement-Learning
```
dann führe aus:
```
pip install -r requirements.txt
```
im installierten `Parallel-Reinforcement-Learning`-Ordner.

## Schnellstart
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

	async_ppo.ppo.save_weights(path='(Ihr Pfad hier)/PPO_PRL/PPO/data/')
```

`PPO`-Parameter:

- `is_continuous` - auf True setzen, wenn die Umgebung kontinuierliche Aktionen erfordert (False bedeutet diskrete Aktionen und True bedeutet kontinuierliche).
- `action_dim`  - Anzahl möglicher Aktionen (z. B. `action_dim=2` für CartPole-v1 oder `action_dim=23` für Pusher-v5).
- `observ_dim` - Anzahl der Zustandsmerkmale (z. B. `observ_dim=4` für CartPole-v1 oder `observ_dim=348` für Humanoid-v5).
- `Actor_lr` - Wert von lr für das Actor-Netzwerk.
- `Critic_lr` - Wert von lr für das Critic-Netzwerk.
- `action_scaling` - Multiplikator für Aktionen, z. B. für Pusher-v5 müssen wir `action_scaling=2.0` verwenden, da der Aktionsbereich in Pusher-v5 (-2, 2) beträgt und unser Netzwerk nur Aktionen von (-1, 1) ausgibt, wenn `is_continuous=True` ist. Daher wird `action_scaling` verwendet, um die Aktionen in den richtigen Bereich zu skalieren.
- `policy_clip` - Wert für Richtlinienänderungen, z. B. `policy_clip=0.2` erlaubt Änderungen von nicht mehr als 20%.
- `k_epochs` - Anzahl der Epochen für das Netzwerk-Lernen auf einem Datensatz.
- `GAE_lambda` - Glättungsfaktor für die Advantage-Berechnung (0 = hohe Varianz, 1 = geringere Varianz).
- `batch_size` - Batch-Größe.
- `mini_batch_size` - Mini-Batch-Größe.
- `gamma` - beeinflusst die Berücksichtigung langfristiger Belohnungen (normalerweise 0.99-0.999).
- `use_RND` - ob wir `Random Network Distillation` verwenden werden.
- `beta` - Multiplikator für `RND`-Belohnungen.

Mehr über RND - https://openai.com/index/reinforcement-learning-with-prediction-based-rewards/

## Benutzerdefinierte Schleife

**WARNUNG**: Die Zeile `env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)` ist äußerst wichtig. Ihr Fehlen wird das Lernen unterbrechen.

Auch in diesem Beispiel werde ich meine eigene Implementierung verwenden, aber Sie können diesen Code gerne kopieren und für Ihre Implementierung umschreiben.

```python
# Importieren
from PPO import PPO
from AsyncPPO import EnvVectorizer, VecMemory

import utils

import torch as t
import numpy as np

from tqdm import tqdm

import gymnasium as gym

# Hauptschleife
if __name__ == '__main__':
	# Umgebung erstellen
	env = gym.make('CartPole-v1')

	# Neuronales Netzwerk initialisieren (oder Ihre eigene Implementierung)
	ppo = PPO(
		is_continuous=False, action_dim=env.action_space.n, observ_dim=env.observation_space.shape[0],
		Actor_lr=0.0010, Critic_lr=0.0025,# action_scaling=1.0,
		policy_clip=0.2, k_epochs=11, GAE_lambda=0.95, 
		batch_size=1024, mini_batch_size=1024, gamma=0.995,
		use_RND=True, beta=0.01
	)

	env = EnvVectorizer(env=env, num_envs=32) # Vektorisierte Umgebung
	buffer = VecMemory(num_envs=32) # Vektorisierter Puffer mit einem Puffer für jede Umgebung

	# Datensammelschleife mit tqdm-Fortschrittsbalken
	for _ in (pbar := tqdm(range(200))):
		states = env.reset()[0] # Zustände abrufen

		rewards_score = np.array(0.) # Belohnungspunktzahl zurücksetzen
		steps_score = np.array(0) # Schrittpunktzahl zurücksetzen

		while True:
			# Aktionen, Zustandswerte und Log-Wahrscheinlichkeiten abrufen
			actions, state_values, log_probs = ppo.get_action(t.from_numpy(states)) 

			# Schritte ausführen
			next_states, rewards, dones, truncates, _ = env.step(actions) 

			# Daten in unseren AsyncPPO.VecMemory-Puffer einfügen
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

			# Zustände mit done oder truncate = True Merkmalen sieben und auch die Aktivitätsliste der Umgebungen aktualisieren
			states = utils.inactive_states_dropout(states, dones | truncates) 
			env.envs_active = utils.update_active_environments_list(env.envs_active, dones | truncates)

			rewards_score += sum(rewards) # Belohnungspunktzahl aktualisieren
			steps_score += len(actions) # Schrittpunktzahl aktualisieren

			# Wenn alle Umgebungen terminal sind, beenden wir die Episode
			if np.all(env.envs_active): 
				# Daten von unserem lokalen Puffer in den ppo.memory-Puffer für das PPO-Lernen übertragen. Sie können auch Ihre eigene Funktion verwenden, um Daten in den Puffer Ihres eigenen neuronalen Netzwerks zu übertragen.
				utils.buffer_to_target_buffer_transfer(buffer, ppo.memory) 
				
				ppo.learn() # Lernfunktion starten

				break # Episode verlassen und eine neue starten
```