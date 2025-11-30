# FILE: generate_sim_data.py
import os
import h5py
import numpy as np
from tqdm import tqdm
from sim_env import SimEnv
from config.config import TASK_CONFIG

# Configuration
cfg = TASK_CONFIG
NUM_EPISODES = 50  # Generate 50 episodes for training
DATA_DIR = os.path.join(cfg['dataset_dir'], 'sim_cube_sort')

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

env = SimEnv()

def scripted_action(t, total_steps, initial_qpos):
    """
    A simple scripted policy:
    1. Move to a target position.
    2. Move back to initial position.
    """
    # Target joint angles (tuned for the XML scene above)
    target_qpos = np.array([0.5, 0.5, -0.5, 0.0]) 
    
    if t < total_steps // 2:
        # Move towards target
        alpha = t / (total_steps // 2)
        action = (1 - alpha) * initial_qpos + alpha * target_qpos
    else:
        # Move back to start
        alpha = (t - total_steps // 2) / (total_steps // 2)
        action = (1 - alpha) * target_qpos + alpha * initial_qpos
        
    return action

print(f"Generating {NUM_EPISODES} episodes of synthetic data...")

for episode_idx in tqdm(range(NUM_EPISODES)):
    env.reset()
    
    obs_replay = []
    action_replay = []
    
    # Get initial state
    initial_obs = env.get_obs()
    initial_qpos = initial_obs['qpos']
    
    # Run Episode
    for t in range(cfg['episode_len']):
        # 1. Observation
        obs = env.get_obs()
        
        # 2. Expert Action (Where should the robot go NEXT?)
        action = scripted_action(t + 1, cfg['episode_len'], initial_qpos)
        
        # 3. Store Data
        obs_replay.append(obs)
        action_replay.append(action)
        
        # 4. Step Environment
        env.step(action)
        
    # Save HDF5
    dataset_path = os.path.join(DATA_DIR, f'episode_{episode_idx}.hdf5')
    
    data_dict = {
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/action': [],
    }
    for cam_name in cfg['camera_names']:
        data_dict[f'/observations/images/{cam_name}'] = []

    for o, a in zip(obs_replay, action_replay):
        data_dict['/observations/qpos'].append(o['qpos'])
        data_dict['/observations/qvel'].append(o['qvel'])
        data_dict['/action'].append(a)
        for cam_name in cfg['camera_names']:
            data_dict[f'/observations/images/{cam_name}'].append(o['images'][cam_name])

    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        for cam_name in cfg['camera_names']:
            _ = image.create_dataset(cam_name, (cfg['episode_len'], cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8', chunks=(1, cfg['cam_height'], cfg['cam_width'], 3))
        
        qpos = obs.create_dataset('qpos', (cfg['episode_len'], 4)) 
        qvel = obs.create_dataset('qvel', (cfg['episode_len'], 4))
        action = root.create_dataset('action', (cfg['episode_len'], 4)) 
        
        for name, array in data_dict.items():
            root[name][...] = array

print("Data generation complete.")