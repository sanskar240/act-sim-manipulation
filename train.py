from config.config import POLICY_CONFIG, TASK_CONFIG, TRAIN_CONFIG
import os
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
import torch
import numpy as np
from training.utils import make_policy, make_optimizer, load_data, compute_dict_mean, detach_dict, set_seed


ACCUMULATION_STEPS = 8  


# parse the task name via command line
parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='sim_cube_sort')
args = parser.parse_args()
task = args.task

# configs
task_cfg = TASK_CONFIG
train_cfg = TRAIN_CONFIG
policy_config = POLICY_CONFIG
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)
device = os.environ['DEVICE']

def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close() # Close figure to save memory

def train_bc(train_dataloader, val_dataloader, policy_config):
    # load policy
    policy = make_policy(policy_config['policy_class'], policy_config)
    policy.to(device)
    optimizer = make_optimizer(policy_config['policy_class'], policy)

    os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    print(f"🚀 Training Started with Gradient Accumulation (Steps={ACCUMULATION_STEPS})")
    print(f"Effective Batch Size: {train_cfg['batch_size_train'] * ACCUMULATION_STEPS}")

    for epoch in range(train_cfg['num_epochs']):
        # --- VALIDATION LOOP ---
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            
            # Save Best Model Logic
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                # Save immediately
                torch.save(policy.state_dict(), os.path.join(checkpoint_dir, 'policy_best.ckpt'))
                print(f"✅ New Best Policy Saved! (Val Loss: {min_val_loss:.4f})")

        # --- TRAINING LOOP (With Gradient Accumulation) ---
        policy.train()
        optimizer.zero_grad()
        epoch_train_stats = []
        
        for batch_idx, data in enumerate(train_dataloader):
            # 1. Forward
            forward_dict = forward_pass(data, policy)
            
            # 2. Normalize Loss (Crucial for Accumulation)
            loss = forward_dict['loss'] / ACCUMULATION_STEPS
            
            # 3. Backward (Accumulate Gradient)
            loss.backward()
            
            # 4. Step Optimizer only every N steps
            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Log the full loss (not divided) for stats
            detached_stats = detach_dict(forward_dict)
            epoch_train_stats.append(detached_stats)

        # Average stats for the epoch
        epoch_summary = compute_dict_mean(epoch_train_stats)
        train_history.append(epoch_summary)
        epoch_train_loss = epoch_summary['loss']

        # Logging
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}')

        # Save Regular Checkpoint every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            ckpt_path = os.path.join(checkpoint_dir, f"policy_epoch_{epoch}_seed_{train_cfg['seed']}.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

    # End of training
    ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    print(f"Training Complete. Best Validation Loss: {min_val_loss:.4f}")

if __name__ == '__main__':
    set_seed(train_cfg['seed'])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Load Data
    data_dir = os.path.join(task_cfg['dataset_dir'], task)
    num_episodes = len([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    print(f"Found {num_episodes} episodes in {data_dir}")

    train_dataloader, val_dataloader, stats, _ = load_data(data_dir, num_episodes, task_cfg['camera_names'],
                                                            train_cfg['batch_size_train'], train_cfg['batch_size_val'])
    
    # Save stats
    stats_path = os.path.join(checkpoint_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    # Force dimensions in config for policy creation
    policy_config['state_dim'] = 4
    policy_config['action_dim'] = 4

    train_bc(train_dataloader, val_dataloader, policy_config)