import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# --- Import your custom modules ---
from models.ae import FullAutoencoder
from data.triplet_dataset import TripletDataset
from core.loss import rc_loss, triplet_loss, cross_rc_loss

class Trainer: # Using the class name from your new code
    """
    A modular class to handle the entire training and evaluation loop.
    """
    def __init__(self, config, train_dataset, val_dataset, model = None):
        """
        Initializes the model, optimizer, loss functions, and data loader.
        (Using your new config structure)
        """
        self.config = config
        self.train_hp = config.get("train_hp") # Fallback to root config
        self.ae_cfg = config.get("ae")       # Fallback to root config
        self.device = torch.device(self.ae_cfg["device"])

        window_size = self.config.get("window", "size")
        self.n_frames = window_size
        self.stride = config.get("window", "stride")
        self.matching = config.get("window", "matching")
        self.n_features = self.ae_cfg["N_FEATURES"]

        self.model_path = f'{self.ae_cfg["full_ae_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
        self.e_m_path = f'{self.ae_cfg["e_m_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'

        print(f"--- Initializing Trainer ---")
        print(f"Using device: {self.device}")

        # 1. Create DataLoaders from the provided datasets
        self.train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.ae_cfg["BATCH_SIZE"],
            shuffle=True,
            num_workers=self.train_hp["NUM_WORKERS"],
            pin_memory=True
        )
        self.patience = self.train_hp.get("es_patience", 10)
        self.min_delta = self.train_hp.get("es_delta", 0.0)

        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.ae_cfg["BATCH_SIZE"],
            shuffle=False, # No need to shuffle validation
            num_workers=self.train_hp["NUM_WORKERS"],
            pin_memory=True
        )
        print(f"Train data loaded: {len(train_dataset)} samples.")
        print(f"Val data loaded: {len(val_dataset)} samples.")
        lr = 0
        # 2. Setup Model
        self.epoch_size = self.train_hp['EPOCHS']
        if model is None:
            self.model = FullAutoencoder(
                input_features=self.ae_cfg["N_FEATURES"],
                num_frames=window_size,
                action_dim=128,
                static_dim=64,
            ).to(self.device)
            lr = self.train_hp["lr"]

        elif isinstance(model, str):

            self.model = torch.load(model, weights_only=False, map_location=self.device)
            lr = self.train_hp["fine_tune_lr"]
            self.epoch_size = (self.train_hp['EPOCHS'] * 2)
        else:
            self.model = model
            lr = self.train_hp["fine_tune_lr"]
            self.epoch_size = (self.train_hp['EPOCHS'] * 2)

        
        # 3. Setup Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=lr
        )

        # 4. Setup Loss Functions
        self.loss_recon_fn = nn.MSELoss()
        self.loss_triplet_fn = nn.TripletMarginLoss(
            margin=self.train_hp["TRIPLET_MARGIN"]
        )

    def _train_step(self, batch):
        """
        Performs a single training step (forward pass, loss, backward pass).
        """
        # --- A. Move data to GPU ---
        for key in batch:
            batch[key] = batch[key].to(self.device)

        # --- B. Clear Gradients ---
        self.optimizer.zero_grad()
        
        # --- C. Encode All Samples ---
        features = {}
        (features["h_a_anchor"], 
         features["h_s_anchor"], 
         features["h_v_anchor"]) = self.model.encode(batch["anchor"])
        
        features["h_a_act_pos"], _, _ = self.model.encode(batch["action_pos"])
        features["h_a_act_neg"], _, _ = self.model.encode(batch["action_neg"])
        
        _, features["h_s_skel_pos"], _ = self.model.encode(batch["skel_pos"])
        _, features["h_s_skel_neg"], _ = self.model.encode(batch["skel_neg"])
        
        _, _, features["h_v_view_pos"] = self.model.encode(batch["view_pos"])
        _, _, features["h_v_view_neg"] = self.model.encode(batch["view_neg"])
        
        # --- D. Decode for Reconstruction ---
        reconstructed_anchor = self.model.decode(
            features["h_a_anchor"], 
            features["h_s_anchor"], 
            features["h_v_anchor"]
        )
        reconstructed_cross = self.model.decode(
            features["h_a_anchor"], 
            features["h_s_skel_neg"], 
            features["h_v_view_neg"]
        )
        
        # --- E. Calculate Losses (Modular) ---
        L_rec = rc_loss(
            reconstructed_anchor, 
            batch["anchor"], 
            self.loss_recon_fn
        )
        
        L_triplet = triplet_loss(features, self.loss_triplet_fn)
        
        L_cross = cross_rc_loss(
            reconstructed_cross, 
            batch["cross_target"], 
            self.loss_recon_fn
        )

        # --- F. Total Loss & Backpropagation ---
        total_loss = L_rec + L_triplet + L_cross
        
        total_loss.backward()
        self.optimizer.step()
        
        # --- G. RETURN ALL LOSSES ---
        return {
            "total": total_loss.item(),
            "rec": L_rec.item(),
            "triplet": L_triplet.item(),
            "cross": L_cross.item()
        }

    def _val_step(self, batch):
        """
        Performs a single validation step.
        (No gradients, no optimizer step)
        """
        # --- A. Move data to GPU ---
        for key in batch:
            batch[key] = batch[key].to(self.device)
        
        # --- B. Encode All Samples ---
        features = {}
        (features["h_a_anchor"], 
         features["h_s_anchor"], 
         features["h_v_anchor"]) = self.model.encode(batch["anchor"])
        
        features["h_a_act_pos"], _, _ = self.model.encode(batch["action_pos"])
        features["h_a_act_neg"], _, _ = self.model.encode(batch["action_neg"])
        
        _, features["h_s_skel_pos"], _ = self.model.encode(batch["skel_pos"])
        _, features["h_s_skel_neg"], _ = self.model.encode(batch["skel_neg"])
        
        _, _, features["h_v_view_pos"] = self.model.encode(batch["view_pos"])
        _, _, features["h_v_view_neg"] = self.model.encode(batch["view_neg"])
        
        # --- C. Decode for Reconstruction ---
        reconstructed_anchor = self.model.decode(
            features["h_a_anchor"], 
            features["h_s_anchor"], 
            features["h_v_anchor"]
        )
        reconstructed_cross = self.model.decode(
            features["h_a_anchor"], 
            features["h_s_skel_neg"], 
            features["h_v_view_neg"]
        )
        
        # --- D. Calculate Losses (Modular) ---
        L_rec = rc_loss(
            reconstructed_anchor, 
            batch["anchor"], 
            self.loss_recon_fn
        )
        L_triplet = triplet_loss(features, self.loss_triplet_fn)
        L_cross = cross_rc_loss(
            reconstructed_cross, 
            batch["cross_target"], 
            self.loss_recon_fn
        )

        # --- E. Total Loss ---
        total_loss = L_rec + L_triplet + L_cross
        
        # --- F. RETURN ALL LOSSES ---
        return {
            "total": total_loss.item(),
            "rec": L_rec.item(),
            "triplet": L_triplet.item(),
            "cross": L_cross.item()
        }

    def fit(self):
            """
            Runs the training loop with Early Stopping.
            """
            print(f"Starting training for {self.epoch_size} epochs...")
            
            # --- EARLY STOPPING STATE ---
            best_val_loss = np.inf
            patience_counter = 0
            
            for epoch in range(self.epoch_size):
                
                # 1. Training Phase
                self.model.train()
                train_losses = {"total": 0.0, "rec": 0.0, "triplet": 0.0, "cross": 0.0}
                
                for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} TRAIN", leave=False):
                    losses = self._train_step(batch)
                    for k in train_losses: train_losses[k] += losses[k]
                
                # Averages
                n_train = len(self.train_loader)
                avg_train = {k: v / n_train for k, v in train_losses.items()}

                # 2. Validation Phase
                self.model.eval()
                val_losses = {"total": 0.0, "rec": 0.0, "triplet": 0.0, "cross": 0.0}
                
                with torch.no_grad():
                    for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} VAL  ", leave=False):
                        losses = self._val_step(batch)
                        for k in val_losses: val_losses[k] += losses[k]
                
                n_val = len(self.val_loader)
                avg_val = {k: v / n_val for k, v in val_losses.items()}

                # 3. Print Stats
                print(f"Epoch [{epoch+1}/{self.epoch_size}]")
                print(f"  Train: Total:{avg_train['total']:.4f} | Rec:{avg_train['rec']:.4f} | Triplet:{avg_train['triplet']:.4f} | Cross:{avg_train['cross']:.4f}")
                print(f"  Val:   Total:{avg_val['total']:.4f} | Rec:{avg_val['rec']:.4f} | Triplet:{avg_val['triplet']:.4f} | Cross:{avg_val['cross']:.4f}")

                # 4. --- EARLY STOPPING CHECK ---
                current_val_loss = avg_val["total"]
                
                # Check if loss improved (must decrease by at least min_delta)
                if current_val_loss < (best_val_loss - self.min_delta):
                    print(f"  [+] Val Loss improved from {best_val_loss:.4f} to {current_val_loss:.4f}. Saving model...")
                    best_val_loss = current_val_loss
                    patience_counter = 0 # Reset counter
                    self.save_model()    # Save the BEST model
                else:
                    patience_counter += 1
                    print(f"  [!] No improvement. Patience: {patience_counter}/{self.patience}")
                    
                    if patience_counter >= self.patience:
                        print(f"--- Early Stopping Triggered at Epoch {epoch+1} ---")
                        print(f"Best Validation Loss: {best_val_loss:.4f}")
                        break

            print("--- Training Finished ---")
            return self.model

    def save_model(self):
        """
        Saves the model in two formats:
        1. Standard .pth (weights only) - useful for resuming training.
        2. JIT .pt (standalone) - useful for deployment/inference without Python class files.
        """
        # --- 1. Save Weights (Standard) ---
        torch.save(self.model, self.model_path)
        torch.save(self.model.E_m, self.e_m_path) # Warning: This line still has file dependencies!
        print(f"Weights saved to {self.model_path}")

        # --- 2. Save JIT (Dependency Free) ---
        
        self.model.eval()
        
        batch = next(iter(self.train_loader))
        for key in batch:
            batch[key] = batch[key].to(self.device)
        data = batch["anchor"]
        try:
            # A. Trace Full Autoencoder
            traced_full = torch.jit.trace(self.model, data)
            jit_full_path = self.model_path.replace(".pth", "_jit.pt")
            traced_full.save(jit_full_path)
            print(f"JIT Full Model saved to {jit_full_path}")

            # B. Trace Action Encoder (E_m) only
            # Assuming E_m takes the same input structure
            traced_enc = torch.jit.trace(self.model.E_m, data.permute(0, 2, 1))
            jit_enc_path = self.e_m_path.replace(".pth", "_jit.pt")
            traced_enc.save(jit_enc_path)
            print(f"JIT Encoder saved to {jit_enc_path}")
            
        except Exception as e:
            print(f"WARNING: JIT Tracing failed. {e}")