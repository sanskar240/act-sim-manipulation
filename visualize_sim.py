import os
import cv2
import torch
import pickle
import numpy as np
from sim_env import SimEnv
from training.utils import make_policy
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG

# --- CONFIGURATION v2 ---
# Set to True to see Silky Smooth Motion (v2)
# Set to False to see Jittery Motion (v1)
ENABLE_TEMPORAL_AGGREGATION = True 

# Smoothing Factor (0.01 is standard). 
# Higher = Trusts recent predictions more. Lower = Smoother but more lag.
EXP_WEIGHT_K = 0.01 
# ------------------------

cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

def load_model_and_stats():
    # Force dimensions
    policy_config['state_dim'] = 4
    policy_config['action_dim'] = 4
    
    # Load Best Checkpoint
    ckpt_name = train_cfg['eval_ckpt_name'] # e.g., 'policy_best.ckpt'
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', ckpt_name)
    stats_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', 'dataset_stats.pkl')

    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found: {ckpt_path}")
        exit()

    print(f"Loading Model: {ckpt_name}")
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.to(device)
    policy.eval()

    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    
    return policy, stats

def main():
    policy, stats = load_model_and_stats()
    env = SimEnv()
    
    # UNSEEN DATA: This resets the cube to a random location
    env.reset() 
    
    # Buffers for Temporal Aggregation
    max_timesteps = 400
    chunk_size = 100 # From config (num_queries)
    action_dim = 4
    
    # This matrix stores ALL predictions ever made
    # Shape: [Time, Time + Chunk_Size, Action_Dim]
    all_time_actions = torch.zeros([max_timesteps, max_timesteps + chunk_size, action_dim]).to(device)
    
    obs = env.get_obs()
    video_frames = []
    
    print(f"Starting Inference (Smoothing: {ENABLE_TEMPORAL_AGGREGATION})...")
    
    for t in range(max_timesteps):
        # 1. Preprocess Observation
        qpos = (obs['qpos'] - stats['qpos_mean']) / stats['qpos_std']
        img = obs['images']['front']
        video_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        qpos_t = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
        img_t = torch.from_numpy(img / 255.0).float().to(device).permute(2,0,1).unsqueeze(0)
        
        # 2. Get Prediction (Chunk of 100 steps)
        with torch.inference_mode():
            action_chunk = policy(qpos_t, img_t) # Shape [1, 100, 4]
        
        if ENABLE_TEMPORAL_AGGREGATION:
            # --- THE MAGIC SAUCE (Temporal Aggregation) ---
            
            # A. Store the prediction in the Time Matrix
            # We place this chunk starting at time 't'
            all_time_actions[t, t : t + chunk_size] = action_chunk
            
            # B. Extract overlap for the CURRENT step 't'
            # We look at the column 't' across all previous rows
            raw_actions = all_time_actions[:, t] # Shape [Max_Steps, 4]
            
            # C. Calculate Weights (Exponential Decay)
            # Only consider actions that are non-zero (populated)
            # Calculate how "old" each prediction is (i)
            # Weight = exp(-k * i)
            valid_mask = (raw_actions.abs().sum(dim=1) > 0)
            valid_actions = raw_actions[valid_mask]
            
            # Create weights based on how many steps ago the prediction was made
            # Example: [exp(-0), exp(-0.01), exp(-0.02)...]
            num_valid = valid_actions.shape[0]
            k = EXP_WEIGHT_K
            exp_weights = np.exp(-k * np.arange(num_valid))
            exp_weights = torch.from_numpy(exp_weights).float().to(device).unsqueeze(1)
            
            # Normalize weights so they sum to 1
            exp_weights = exp_weights / exp_weights.sum()
            
            # D. Weighted Average
            final_action = (valid_actions * exp_weights).sum(dim=0)
            final_action = final_action.cpu().numpy()
            
        else:
            # --- THE OLD WAY (v1 Jittery) ---
            # Just take the first step of the current plan and ignore history
            final_action = action_chunk[0, 0].cpu().numpy()

        # 3. Post-process and Execute
        real_action = final_action * stats['action_std'] + stats['action_mean']
        env.step(real_action)
        obs = env.get_obs()
        
        if t % 50 == 0: print(f"Step {t}/{max_timesteps}")

    # Save Video
    filename = "sim_result_smooth.mp4" if ENABLE_TEMPORAL_AGGREGATION else "sim_result_jitter.mp4"
    h, w, _ = video_frames[0].shape
    out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
    for f in video_frames: out.write(f)
    out.release()
    print(f"✅ Video Saved: {filename}")

if __name__ == "__main__":
    main()