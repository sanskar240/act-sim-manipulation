import os
import h5py
import numpy as np
from tqdm import tqdm
from sim_env import SimEnv
from config.config import TASK_CONFIG

cfg = TASK_CONFIG
NUM_EPISODES = 300
DATA_DIR = os.path.join(cfg['dataset_dir'], 'sim_cube_sort')

if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

env = SimEnv()

def generate_random_target_qpos():
    j1 = np.random.uniform(-1.0, 1.0)
    j2 = np.random.uniform(0.0, 1.0) 
    j3 = np.random.uniform(-1.0, 0.0)
    j4 = np.random.uniform(-0.5, 0.5)
    return np.array([j1, j2, j3, j4])

def generate_random_home_qpos():
    return np.random.uniform(-0.2, 0.2, size=4)

print(f"Generating {NUM_EPISODES} noisy episodes (Teacher v2)...")

for episode_idx in tqdm(range(NUM_EPISODES)):
    env.reset()
    target_qpos = generate_random_target_qpos()
    
    # Teleport to find hand position
    env.step(target_qpos) 
    cube_target_pos = env.get_ee_pos()
    
    # Set cube and reset robot
    env.set_cube_pos(cube_target_pos)
    home_qpos = generate_random_home_qpos()
    env.step(home_qpos)
    
    obs_replay = []
    action_replay = []
    total_steps = cfg['episode_len']
    
    for t in range(total_steps):
        obs = env.get_obs()
        alpha = (t + 1) / total_steps
        ideal_action = (1 - alpha) * home_qpos + alpha * target_qpos
        # Inject Noise
        noise = np.random.normal(0, 0.01, size=4)
        action = ideal_action + noise
        
        obs_replay.append(obs)
        action_replay.append(action)
        env.step(action)
        
    dataset_path = os.path.join(DATA_DIR, f'episode_{episode_idx}.hdf5')
    data_dict = {
        '/observations/qpos': [], '/observations/qvel': [], '/action': [],
        '/observations/images/front': []
    }

    for o, a in zip(obs_replay, action_replay):
        data_dict['/observations/qpos'].append(o['qpos'])
        data_dict['/observations/qvel'].append(o['qvel'])
        data_dict['/action'].append(a)
        data_dict['/observations/images/front'].append(o['images']['front'])

    with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
        root.attrs['sim'] = True
        obs = root.create_group('observations')
        image = obs.create_group('images')
        _ = image.create_dataset('front', (total_steps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8', chunks=(1, cfg['cam_height'], cfg['cam_width'], 3))
        qpos = obs.create_dataset('qpos', (total_steps, 4)) 
        qvel = obs.create_dataset('qvel', (total_steps, 4))
        action = root.create_dataset('action', (total_steps, 4)) 
        for name, array in data_dict.items(): root[name][...] = array