import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tqdm import tqdm

# --- Import your custom modules ---
from models.ae import FullAutoencoder
from data.triplet_dataset import TripletDataset
from core.loss import rc_loss, triplet_loss, cross_rc_loss

class Trainer: # Using the class name from your new code
    """
    A modular class to handle the entire training and evaluation loop.
    """
    def __init__(self, config, train_dataset, val_dataset):
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
        self.val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=self.ae_cfg["BATCH_SIZE"],
            shuffle=False, # No need to shuffle validation
            num_workers=self.train_hp["NUM_WORKERS"],
            pin_memory=True
        )
        print(f"Train data loaded: {len(train_dataset)} samples.")
        print(f"Val data loaded: {len(val_dataset)} samples.")

        # 2. Setup Model
        self.model = FullAutoencoder(
            input_features=self.ae_cfg["N_FEATURES"],
            num_frames=window_size,
            action_dim=self.ae_cfg["ACTION_DIM"],
            static_dim=self.ae_cfg["STATIC_DIM"]
        ).to(self.device)

        # 3. Setup Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.train_hp["lr"]
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
        Runs the full training loop for all epochs.
        (This now includes the validation loop)
        """
        print(f"Starting training for {self.train_hp['EPOCHS']} epochs...")
        
        for epoch in range(self.train_hp['EPOCHS']):
            
            # --- TRAINING LOOP ---
            self.model.train()
            train_losses = {"total": 0.0, "rec": 0.0, "triplet": 0.0, "cross": 0.0}
            
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1} TRAIN", leave=False):
                losses = self._train_step(batch)
                # --- Accumulate all losses ---
                train_losses["total"] += losses["total"]
                train_losses["rec"] += losses["rec"]
                train_losses["triplet"] += losses["triplet"]
                train_losses["cross"] += losses["cross"]
            
            # --- Calculate average training losses ---
            num_train_batches = len(self.train_loader)
            avg_train_total = train_losses["total"] / num_train_batches
            avg_train_rec = train_losses["rec"] / num_train_batches
            avg_train_triplet = train_losses["triplet"] / num_train_batches
            avg_train_cross = train_losses["cross"] / num_train_batches


            # --- VALIDATION LOOP ---
            self.model.eval() # Set model to evaluation mode
            # --- Initialize loss accumulators ---
            val_losses = {"total": 0.0, "rec": 0.0, "triplet": 0.0, "cross": 0.0}
            
            with torch.no_grad(): # Disable gradient calculation
                for batch in tqdm(self.val_loader, desc=f"Epoch {epoch+1} VAL  ", leave=False):
                    losses = self._val_step(batch)
                    # --- Accumulate all losses ---
                    val_losses["total"] += losses["total"]
                    val_losses["rec"] += losses["rec"]
                    val_losses["triplet"] += losses["triplet"]
                    val_losses["cross"] += losses["cross"]
            
            # --- Calculate average validation losses ---
            num_val_batches = len(self.val_loader)
            avg_val_total = val_losses["total"] / num_val_batches
            avg_val_rec = val_losses["rec"] / num_val_batches
            avg_val_triplet = val_losses["triplet"] / num_val_batches
            avg_val_cross = val_losses["cross"] / num_val_batches


            print(f"Epoch [{epoch+1}/{self.train_hp['EPOCHS']}]")
            print(f"  Train Loss: {avg_train_total:.4f} | "
                  f"L_rec: {avg_train_rec:.4f} | "
                  f"L_triplet: {avg_train_triplet:.4f} | "
                  f"L_cross: {avg_train_cross:.4f}")
            print(f"  Val Loss:   {avg_val_total:.4f} | "
                  f"L_rec: {avg_val_rec:.4f} | "
                  f"L_triplet: {avg_val_triplet:.4f} | "
                  f"L_cross: {avg_val_cross:.4f}")

        print("--- Training Finished ---")
        self.save_model()

    def save_model(self):
        """Saves the trained Action Encoder (E_m) weights."""
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Full Autoencoder saved to {self.model_path}")
        
        # You can still save the E_m separately if you want
        torch.save(self.model.E_m, self.e_m_path)
        print(f"Action Encoder (E_m) saved to {self.e_m_path}")