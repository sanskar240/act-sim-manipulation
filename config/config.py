import os
import torch

DATA_DIR = 'data/'
CHECKPOINT_DIR = 'checkpoints/'
device = 'cpu'
if torch.cuda.is_available(): device = 'cuda'
if torch.backends.mps.is_available(): device = 'mps'
os.environ['DEVICE'] = device

ROBOT_PORTS = {'leader': '/dev/null', 'follower': '/dev/null'}

TASK_CONFIG = {
    'dataset_dir': DATA_DIR,
    'episode_len': 300,
    'state_dim': 4,
    'action_dim': 4,
    'cam_width': 640,
    'cam_height': 480,
    'camera_names': ['front'],
    'camera_port': 0
}

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
    'state_dim': 4,
    'action_dim': 4
}

TRAIN_CONFIG = {
    'seed': 42,
    'num_epochs': 500,     # INCREASED
    'batch_size_val': 4,
    'batch_size_train': 4,
    'eval_ckpt_name': 'policy_best.ckpt', # LOOK FOR BEST
    'checkpoint_dir': CHECKPOINT_DIR
}