from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
import os
import pickle
import argparse
from copy import deepcopy
import torch
import numpy as np
import zipfile
from huggingface_hub import hf_hub_download
from training.utils import make_policy, make_optimizer, load_data, compute_dict_mean, detach_dict, set_seed

# --- M2 OPTIMIZATION ---
ACCUMULATION_STEPS = 8  # Virtual Batch Size = 32
# -----------------------

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sim_cube_sort')
args = parser.parse_args()
task = args.task

task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
device = os.environ['DEVICE']

def setup_data():
    """Downloads and unzips data from Hugging Face if needed."""
    data_path = os.path.join(task_cfg['dataset_dir'], task)
    
    if not os.path.exists(data_path) or len(os.listdir(data_path)) == 0:
        print(f"📦 Downloading dataset from sanskxr02/act_sim_cube_sort...")
        try:
            zip_path = hf_hub_download(
                repo_id="sanskxr02/act_sim_cube_sort",
                filename="episodes-v2.zip",
                repo_type="dataset"
            )
            print("📦 Unzipping...")
            os.makedirs(data_path, exist_ok=True)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_path)
            
            # Move files up if in subfolder
            hdf5 = [f for f in os.listdir(data_path) if f.endswith('.hdf5')]
            if len(hdf5) == 0:
                subfolders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
                if subfolders:
                    import shutil
                    src = os.path.join(data_path, subfolders[0])
                    for f in os.listdir(src):
                        shutil.move(os.path.join(src, f), data_path)
                    os.rmdir(src)
            print("✅ Data ready.")
        except Exception as e:
            print(f"❌ Failed to download: {e}")
            exit()
    return data_path

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    return policy(qpos_data.to(device), image_data.to(device), action_data.to(device), is_pad.to(device))

def train_bc(train_dataloader, val_dataloader, policy_config):
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_config['policy_class'], policy)
    os.makedirs(checkpoint_dir, exist_ok=True)

    min_val_loss = np.inf
    print(f"🚀 Training v2 (Accum={ACCUMULATION_STEPS}, Epochs={train_cfg['num_epochs']})")

    for epoch in range(train_cfg['num_epochs']):
        # Validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            epoch_val_loss = epoch_summary['loss']
            
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                torch.save(policy.state_dict(), os.path.join(checkpoint_dir, 'policy_best.ckpt'))
                print(f"✅ Epoch {epoch}: New Best Policy! Val Loss: {min_val_loss:.4f}")

        # Training
        policy.train()
        optimizer.zero_grad()
        epoch_train_stats = []
        
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss'] / ACCUMULATION_STEPS
            loss.backward()
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            epoch_train_stats.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(epoch_train_stats)
        if epoch % 10 == 0:
            print(f'   Epoch {epoch}: Train Loss: {epoch_summary["loss"]:.4f}')

    ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    print("Training Complete.")

if __name__ == '__main__':
    set_seed(train_cfg['seed'])
    
    # 1. Download & Setup Data
    data_dir = setup_data()

    # 2. Load Data
    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    print(f"Found {num_episodes} episodes.")
    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'], train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    
    # 3. Save Stats
    with open(os.path.join(checkpoint_dir, f'dataset_stats.pkl'), 'wb') as f: pickle.dump(stats, f)
    
    # 4. Train
    train_bc(train_dataloader, val_dataloader, policy_config)