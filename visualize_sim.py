import os
import cv2
import torch
import pickle
import numpy as np
from sim_env import SimEnv
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
from training.utils import make_policy

# --- CONFIG ---
ENABLE_TEMPORAL_AGGREGATION = True
EXP_WEIGHT_K = 0.01 
# --------------

cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

def load_model():
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', train_cfg['eval_ckpt_name'])
    stats_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', 'dataset_stats.pkl')

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: {ckpt_path} not found. Train first!")
        exit()

    print(f"Loading: {ckpt_path}")
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()

    with open(stats_path, 'rb') as f: stats = pickle.load(f)
    return policy, stats

def main():
    policy, stats = load_model()
    env = SimEnv()
    env.reset() # Unseen Data
    
    # Temporal Aggregation Buffer
    max_steps = 400
    chunk_size = 100
    all_time_actions = torch.zeros([max_steps, max_steps + chunk_size, 4]).to(device)
    
    obs = env.get_obs()
    video_frames = []
    
    print(f"Running Inference (Smoothing={ENABLE_TEMPORAL_AGGREGATION})...")

    for t in range(max_steps):
        # 1. Prepare Input
        qpos = (obs['qpos'] - stats['qpos_mean']) / stats['qpos_std']
        img = obs['images']['front'] / 255.0
        video_frames.append(cv2.cvtColor(obs['images']['front'], cv2.COLOR_RGB2BGR))

        qpos_t = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        img_t = torch.from_numpy(img).float().to(device).permute(2,0,1).unsqueeze(0).unsqueeze(1) # Add Cam dim

        # 2. Inference
        with torch.inference_mode():
            action_chunk = policy(qpos_t, img_t) # [1, 100, 4]

        # 3. Temporal Aggregation
        if ENABLE_TEMPORAL_AGGREGATION:
            all_time_actions[t, t:t+chunk_size] = action_chunk
            raw_actions = all_time_actions[:, t] # Actions for *this* step from all past plans
            
            # Filter valid (non-zero) actions
            valid_mask = (raw_actions.abs().sum(dim=1) > 0)
            valid_actions = raw_actions[valid_mask]
            
            # Weighted Average
            k_weights = np.exp(-EXP_WEIGHT_K * np.arange(len(valid_actions)))
            k_weights = torch.from_numpy(k_weights).float().to(device).unsqueeze(1)
            final_action = (valid_actions * k_weights).sum(dim=0) / k_weights.sum()
            final_action = final_action.cpu().numpy()
        else:
            final_action = action_chunk[0, 0].cpu().numpy()

        # 4. Step
        real_action = final_action * stats['action_std'] + stats['action_mean']
        env.step(real_action)
        obs = env.get_obs()
        
        if t % 50 == 0: print(f"Step {t}")

    # Save Video
    out = cv2.VideoWriter('sim_result_v2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))
    for f in video_frames: out.write(f)
    out.release()
    print("✅ Video saved: sim_result_v2.mp4")

if __name__ == "__main__":
    main()