# FILE: visualize_sim.py
import torch
import cv2
import os
import pickle
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sim_env import SimEnv
from training.utils import make_policy
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG

# Configuration
cfg = TASK_CONFIG
policy_config = POLICY_CONFIG
train_cfg = TRAIN_CONFIG
device = os.environ['DEVICE']

# 1. Load the "Brain" (Policy)
def load_model():
    # Force dimensions again just to be safe
    policy_config['state_dim'] = 4
    policy_config['action_dim'] = 4
    
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', 'policy_last.ckpt')
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        print("Did you finish training?")
        exit()

    print(f"Loading model from: {ckpt_path}")
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    policy.to(device)
    policy.eval()
    return policy

# 2. Load the Statistics (To un-normalize data)
def load_stats():
    stats_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    return stats

def main():
    policy = load_model()
    stats = load_stats()
    env = SimEnv()
    
    # Pre-processing functions (Normalization)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    print("Starting Inference Simulation...")
    env.reset()
    
    # Lists to store video frames and data
    video_frames = []
    qpos_history = []
    target_history = []

    # Get initial observation
    obs = env.get_obs()
    
    # Run for specific length
    max_steps = 300
    for t in range(max_steps):
        # A. Prepare Input for AI
        qpos_numpy = obs['qpos']
        qpos = pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos).float().to(device).unsqueeze(0) # Shape: [1, 4]
        
        # Prepare Image
        img = obs['images']['front']
        # Save frame for video (Convert RGB to BGR for OpenCV)
        video_frames.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        # Reshape image for AI: [H, W, C] -> [1, C, H, W] and normalize
        curr_image = torch.from_numpy(img / 255.0).float().to(device)
        curr_image = curr_image.permute(2, 0, 1).unsqueeze(0)

        # B. Ask AI for Action
        with torch.inference_mode():
            # The model returns a chunk of actions (e.g., 100 steps)
            # We just take the first one for this simple test (No Temporal Aggregation for simplicity)
            action_chunk = policy(qpos, curr_image) 
            next_action_raw = action_chunk[:, 0, :] # Take step 0
            
        # C. Un-normalize Action
        next_action = post_process(next_action_raw.cpu().numpy()[0])
        
        # D. Move Robot
        env.step(next_action)
        obs = env.get_obs()
        
        # Log data
        qpos_history.append(qpos_numpy)
        target_history.append(next_action)
        
        if t % 50 == 0:
            print(f"Step {t}/{max_steps}")

    # 3. Save Video
    output_video = 'sim_result.mp4'
    height, width, layers = video_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video = cv2.VideoWriter(output_video, fourcc, 30, (width, height))

    for frame in video_frames:
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    print(f"\n✅ Video saved to: {os.path.abspath(output_video)}")
    
    # 4. Plot Joints (Optional)
    qpos_history = np.array(qpos_history)
    target_history = np.array(target_history)
    plt.figure(figsize=(10, 5))
    plt.plot(qpos_history[:, 0], label='Joint 0 Actual')
    plt.plot(target_history[:, 0], label='Joint 0 Target', linestyle='--')
    plt.legend()
    plt.title("Robot Movement: Actual vs AI Target")
    plt.savefig('joint_plot.png')
    print("✅ Plot saved to: joint_plot.png")

if __name__ == "__main__":
    main()