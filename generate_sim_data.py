import os
import h5py
import numpy as np
from tqdm import tqdm
from sim_env import SimEnv
from config.config import TASK_CONFIG

# --- CONFIG V2 ---
cfg = TASK_CONFIG
NUM_EPISODES = 300  # Increased from 50 -> 300
DATA_DIR = os.path.join(cfg['dataset_dir'], 'sim_cube_sort')
# -----------------

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

env = SimEnv()

def generate_random_target_qpos():
    """Generates a random joint configuration that is visibly reachable."""
    # Joint limits: [-3, 3]
    # We constrain them to keep the arm in front of the camera
    j1 = np.random.uniform(-1.0, 1.0) # Base rotation
    j2 = np.random.uniform(0.0, 1.0)  # Shoulder
    j3 = np.random.uniform(-1.0, 0.0) # Elbow
    j4 = np.random.uniform(-0.5, 0.5) # Wrist
    return np.array([j1, j2, j3, j4])

def generate_random_home_qpos():
    """Generates a random START position (so robot doesn't always start at 0)"""
    return np.random.uniform(-0.2, 0.2, size=4)

print(f"Generating {NUM_EPISODES} noisy episodes (Teacher v2)...")

for episode_idx in tqdm(range(NUM_EPISODES)):
    env.reset()
    
    # 1. Determine the Goal (Target)
    target_qpos = generate_random_target_qpos()
    
    # 2. Teleport Robot to Goal to find out where the hand ends up
    env.step(target_qpos) 
    cube_target_pos = env.get_ee_pos()
    
    # 3. Teleport Cube to that Hand Position
    env.set_cube_pos(cube_target_pos)
    
    # 4. Reset Robot to a Random Start Position
    home_qpos = generate_random_home_qpos()
    env.step(home_qpos) # Move robot to start
    
    # 5. Record the Episode (Home -> Target)
    obs_replay = []
    action_replay = []
    
    # Interpolation Loop
    total_steps = cfg['episode_len']
    
    for t in range(total_steps):
        # A. Get Observation
        obs = env.get_obs()
        
        # B. Calculate Teacher Action (Linear Interpolation)
        # We add tiny noise to the path so it's not perfectly straight lines
        alpha = (t + 1) / total_steps
        
        # Pure Interpolation
        ideal_action = (1 - alpha) * home_qpos + alpha * target_qpos
        
        # Inject tiny "motor noise" (Teacher Imperfection)
        # This makes the model robust to small errors
        noise = np.random.normal(0, 0.01, size=4)
        action = ideal_action + noise
        
        # C. Store
        obs_replay.append(obs)
        action_replay.append(action)
        
        # D. Step Env
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
            _ = image.create_dataset(cam_name, (total_steps, cfg['cam_height'], cfg['cam_width'], 3), dtype='uint8', chunks=(1, cfg['cam_height'], cfg['cam_width'], 3))
        
        qpos = obs.create_dataset('qpos', (total_steps, 4)) 
        qvel = obs.create_dataset('qvel', (total_steps, 4))
        action = root.create_dataset('action', (total_steps, 4)) 
        
        for name, array in data_dict.items():
            root[name][...] = array

print("Data generation complete.")