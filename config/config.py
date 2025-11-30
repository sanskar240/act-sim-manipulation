import os
# fallback to cpu if mps is not available for specific operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
import torch

# data directory
DATA_DIR = 'data/'

# checkpoint directory
CHECKPOINT_DIR = 'checkpoints/'

# device
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
if torch.backends.mps.is_available(): device = 'mps'  # <-- UNCOMMENT THIS LINE!
os.environ['DEVICE'] = device

# robot port names
ROBOT_PORTS = {
    'leader': '/dev/tty.usbmodem57380045221',
    'follower': '/dev/tty.usbmodem57380046991'
}



TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 300,
    'state_dim': 4,       # CHANGED: Was 5, now 4 (matches SimEnv)
    'action_dim': 4,      # CHANGED: Was 5, now 4 (matches SimEnv)
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'], # CHANGED: Ensure this matches SimEnv camera name
    'camera_port': 0
}


# policy config
# config/config.py

# ... existing code ...

# policy config
POLICY_CONFIG = {
    'lr': 1e-5,
    'device': device,
    'num_queries': 100,
    'kl_weight': 10,
    'hidden_dim': 512,
    'dim_feedforward': 3200,
    'lr_backbone': 1e-5,
    'backbone': 'resnet18',
    'enc_layers': 4,
    'dec_layers': 7,
    'nheads': 8,
    'camera_names': ['front'],
    'policy_class': 'ACT',
    'temporal_agg': False,
    'state_dim': 4,   # <--- ADD THIS (Must match your data)
    'action_dim': 4   # <--- ADD THIS (Must match your data)
}

# ... existing code ...

# training config
TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 2000,
    'batch_size_val': 4,
    'batch_size_train': 4,
    'eval_ckpt_name': 'policy_last.ckpt',
    'checkpoint_dir': CHECKPOINT_DIR
}