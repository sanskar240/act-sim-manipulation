import os
import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
import h5py

# --- CRITICAL FIX: Import Config FIRST to set up 'DEVICE' environment variable ---
from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
# ---------------------------------------------------------------------------------

from training.utils import make_policy

# --- CONFIGURATION ---
# We use Episode 45 (Validation Set)
TEST_EPISODE_IDX = 45 
# ---------------------

def main():
    # 1. Setup & Load Configs
    cfg = TASK_CONFIG
    policy_config = POLICY_CONFIG
    train_cfg = TRAIN_CONFIG
    device = os.environ['DEVICE'] # Now this will work!

    # Force dimensions to match your Sim (4 joints)
    policy_config['state_dim'] = 4
    policy_config['action_dim'] = 4

    # 2. Locate Files
    ckpt_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', train_cfg['eval_ckpt_name'])
    stats_path = os.path.join(train_cfg['checkpoint_dir'], 'sim_cube_sort', 'dataset_stats.pkl')

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint not found at {ckpt_path}")
        print("Check config.py -> 'eval_ckpt_name'")
        return

    # 3. Load The Brain
    print(f"Loading Model from: {ckpt_path}")
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    policy.to(device)
    policy.eval()

    # 4. Load The Glasses (Stats)
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # 5. Load The Unseen Test Data
    dataset_path = os.path.join(cfg['dataset_dir'], 'sim_cube_sort', f'episode_{TEST_EPISODE_IDX}.hdf5')
    print(f"Testing on Unseen Data: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Data file not found. Please check if 'episode_{TEST_EPISODE_IDX}.hdf5' exists in data/sim_cube_sort/")
        return

    with h5py.File(dataset_path, 'r') as root:
        qpos_gt = root['/observations/qpos'][()]
        image_gt = root['/observations/images/front'][()]
        action_gt = root['/action'][()]

    # 6. Run Inference at a specific time step (e.g., t=100)
    t = 100
    
    # Prepare inputs (Normalize)
    qpos = (qpos_gt[t] - stats['qpos_mean']) / stats['qpos_std']
    image = image_gt[t] / 255.0
    
    # Convert to Tensor
    qpos_tensor = torch.from_numpy(qpos).float().to(device).unsqueeze(0)
    image_tensor = torch.from_numpy(image).float().to(device).permute(2, 0, 1).unsqueeze(0)

    print("Asking Model for trajectory...")
    with torch.inference_mode():
        # Model returns [1, 100, 4]
        action_chunk = policy(qpos_tensor, image_tensor)
    
    # Un-normalize output
    pred_action = action_chunk.squeeze(0).cpu().numpy()
    pred_action = pred_action * stats['action_std'] + stats['action_mean']

    # 7. Visualization
    print("Generating Plot...")
    plt.figure(figsize=(15, 5))

    # Plot Joint 0
    plt.subplot(1, 3, 1)
    plt.plot(action_gt[t:t+100, 0], 'k--', label='Expert (Truth)', linewidth=2)
    plt.plot(pred_action[:, 0], 'r', label='ACT Model', linewidth=2)
    plt.title("Joint 0 (Base)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot Joint 1
    plt.subplot(1, 3, 2)
    plt.plot(action_gt[t:t+100, 1], 'k--', label='Expert', linewidth=2)
    plt.plot(pred_action[:, 1], 'g', label='ACT Model', linewidth=2)
    plt.title("Joint 1 (Shoulder)")
    plt.grid(True, alpha=0.3)

    # Plot Joint 2
    plt.subplot(1, 3, 3)
    plt.plot(action_gt[t:t+100, 2], 'k--', label='Expert', linewidth=2)
    plt.plot(pred_action[:, 2], 'b', label='ACT Model', linewidth=2)
    plt.title("Joint 2 (Elbow)")
    plt.grid(True, alpha=0.3)

    plt.suptitle(f"Model Prediction vs Expert on UNSEEN Episode {TEST_EPISODE_IDX}")
    plt.tight_layout()
    plt.savefig('prediction_analysis.png')
    print("✅ Result saved to: prediction_analysis.png")

if __name__ == '__main__':
    main()