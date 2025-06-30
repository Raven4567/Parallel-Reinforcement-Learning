import numpy as np

def indexes_of_active_environments(num_envs: int, is_env_terminal: np.ndarray):
    return np.arange(num_envs)[~is_env_terminal]

def number_of_active_environments(is_env_terminal: np.ndarray):
    return np.sum(~is_env_terminal)

def range_of_active_environments(is_env_terminal: np.ndarray):
    return np.arange(
        number_of_active_environments(is_env_terminal)
    )

def inactive_states_dropout(states: np.ndarray, dones: np.ndarray):
    return states[~dones]

def buffer_append(
        # buffer (usually multiple lists for each env)
        buffer,

        # env data
        states: np.ndarray, 
        actions: np.ndarray, 
        rewards: np.ndarray, 
        dones: np.ndarray, 

        # envs active
        is_env_terminal: np.ndarray,
        num_envs: int
    ):

    i = range_of_active_environments(is_env_terminal) # 
    idx = indexes_of_active_environments(num_envs, is_env_terminal)

    for i_, idx_ in zip(i, idx):
        buffer.push(idx_, states[i_], actions[i_], rewards[i_], dones[i_])

def update_active_environments_list(is_env_terminal: np.ndarray, dones: np.ndarray):
    active_indexes = np.where(~is_env_terminal)[0]

    is_env_terminal[active_indexes] = dones
    
    return is_env_terminal

def buffer_to_target_buffer_transfer(buffer, target_buffer):
    target_buffer.states += sum(buffer.states, [])
    target_buffer.actions += sum(buffer.actions, [])
    target_buffer.rewards += sum(buffer.rewards, [])
    target_buffer.dones += sum(buffer.dones, [])

    buffer.clear()