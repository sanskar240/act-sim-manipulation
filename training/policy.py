import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
import torch

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')
        
        # --- ADAPTER FIX START ---
        # Detect what the library actually built vs what our data is
        self.model_action_dim = self.model.encoder_action_proj.in_features
        self.model_state_dim = self.model.encoder_joint_proj.in_features
        self.data_action_dim = args_override['action_dim']
        
        print(f"\n--- DIMENSION CHECK ---")
        print(f"Data provides: {self.data_action_dim} inputs")
        print(f"Model expects: {self.model_action_dim} inputs")
        if self.model_action_dim != self.data_action_dim:
            print(f"⚠️ MISMATCH DETECTED: Activating auto-padding adapter.")
        else:
            print(f"✅ Dimensions match.")
        print(f"-----------------------\n")
        # --- ADAPTER FIX END ---

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        
        # --- AUTO-PADDING INPUTS ---
        # If model wants 5 but we have 4, pad with zeros
        if qpos.shape[-1] < self.model_state_dim:
            diff = self.model_state_dim - qpos.shape[-1]
            qpos = F.pad(qpos, (0, diff)) # Pad last dimension
            
        if actions is not None and actions.shape[-1] < self.model_action_dim:
            diff = self.model_action_dim - actions.shape[-1]
            actions = F.pad(actions, (0, diff))
        # ---------------------------

        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            
            # --- AUTO-SLICING OUTPUT ---
            # Model outputs 5, we only want to compare the first 4 (the real ones)
            if a_hat.shape[-1] > self.data_action_dim:
                a_hat = a_hat[..., :self.data_action_dim]
                # We also need to slice the 'actions' back to 4 for the loss function
                actions = actions[..., :self.data_action_dim]
            # ---------------------------

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            
            # --- AUTO-SLICING OUTPUT (INFERENCE) ---
            if a_hat.shape[-1] > self.data_action_dim:
                a_hat = a_hat[..., :self.data_action_dim]
            # ---------------------------------------
            
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld