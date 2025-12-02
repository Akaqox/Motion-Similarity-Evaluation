import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp
import warnings
import os
import seaborn as sns


class Demonstrator:
    """
    A class to demonstrate the capabilities of a trained
    FullAutoencoder maturesodel.
    """
    def __init__(self, config):
        self.ae_cfg = config.get("ae")
        self.ws_cfg = config.get("window") # Now correctly reads from config

        self.device = torch.device(self.ae_cfg["device"])
        self.n_frames = self.ws_cfg["size"]
        self.n_features = self.ae_cfg["N_FEATURES"]
        self.POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


    def _plot_skeleton(self, ax, skeleton, color, label):
        """Helper function to plot a single 2D skeleton."""
        # Handle flattening cases first
        if skeleton.ndim == 1:
            if skeleton.shape[0] == 66:
                skeleton = skeleton.reshape(33, 2)
            elif skeleton.shape[0] == 99:
                skeleton = skeleton.reshape(33, 3)
        
        # If 3D (33, 3), project to 2D by taking first 2 cols
        if skeleton.shape == (33, 3):
            skeleton = skeleton[:, :2]

        # Final check
        if skeleton.shape != (33, 2):
            raise ValueError(f"Expected skeleton shape (33, 2) or (33, 3), but got {skeleton.shape}")
            
        x_coords = skeleton[:, 0]
        y_coords = skeleton[:, 1]
        
        # Plot Bones
        for (start_idx, end_idx) in self.POSE_CONNECTIONS:
            if start_idx < 33 and end_idx < 33:
                x_line = [x_coords[start_idx], x_coords[end_idx]]
                y_line = [y_coords[start_idx], y_coords[end_idx]]
                line_label = label if start_idx == 0 and end_idx == 1 else None
                ax.plot(x_line, y_line, f'{color}-', alpha=0.7, label=line_label)
        
        # Plot Joints
        ax.scatter(x_coords, y_coords, s=15, c=color, zorder=3)

    def _animate_comparison(self, fig, ax1, ax2, window_batch1, window_batch2, title1, title2):
        """
        Helper to animate two BATCHES of windows side-by-side.
        Inputs: window_batch1, window_batch2 (shape [B, N_FRAMES, F]) where F is 66 or 99
        """
        batch_size = window_batch1.shape[0]
        n_feats = window_batch1.shape[-1]

        # 1. Unified Projection Logic (66->2D, 99->3D->2D)
        def project_to_2d(batch):
            if batch.shape[-1] == 99:
                # Reshape to (B, Frames, 33, 3) then take x,y only
                return batch.reshape(batch_size, self.n_frames, 33, 3)[..., :2]
            elif batch.shape[-1] == 66:
                # Reshape directly to (B, Frames, 33, 2)
                return batch.reshape(batch_size, self.n_frames, 33, 2)
            else:
                raise ValueError(f"Unexpected feature size: {batch.shape[-1]}")

        # frames1/2 are now guaranteed to be (B, F, 33, 2)
        frames1 = project_to_2d(window_batch1)
        frames2 = project_to_2d(window_batch2)
        
        # Find global axis limits based on the projected 2D data
        all_data = np.concatenate((frames1, frames2))
        x_min, x_max = all_data[..., 0].min(), all_data[..., 0].max()
        y_min, y_max = all_data[..., 1].min(), all_data[..., 1].max()
        padding = 0.2

        def set_axes_limits(ax):
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.grid(True)
            ax.axis('equal')
            ax.invert_yaxis() 

        total_animation_frames = batch_size * self.n_frames

        def update(i): 
            ax1.clear()
            ax2.clear()
            
            batch_idx = i // self.n_frames
            frame_idx = i % self.n_frames
            
            self._plot_skeleton(ax1, frames1[batch_idx, frame_idx], 'b', 'Left')
            ax1.set_title(f"{title1} (Window {batch_idx}, Frame {frame_idx})")
            set_axes_limits(ax1)

            self._plot_skeleton(ax2, frames2[batch_idx, frame_idx], 'r', 'Right')
            ax2.set_title(f"{title2} (Window {batch_idx}, Frame {frame_idx})")
            set_axes_limits(ax2)

        ani = FuncAnimation(fig, update, frames=total_animation_frames, interval=100, blit=False, repeat=True)
        return ani

    def run_reconstruction_test(self, model, original_window_batch):
        """
        Tests the model's ability to reconstruct a given window.
        Input: original_window_batch (Tensor, shape [BATCH, N_FRAMES, 33, 2])
        """
        print("Running Reconstruction Test...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        plt.suptitle("Test 1: Reconstruction (Displaying first window in batch)")

        if not isinstance(original_window_batch, torch.Tensor):
            original_window_batch = torch.tensor(original_window_batch, dtype=torch.float32)

        # 2. Flatten features: (B, F, 33, 2) -> (B, F, 66)
        window_batch_tensor = original_window_batch.flatten(start_dim=2).float().to(self.device)
        
        with torch.no_grad():
            reconstructed_batch_tensor = model(window_batch_tensor)
            
        original_np = window_batch_tensor.detach().clone().cpu().numpy()
        reconstructed_np = reconstructed_batch_tensor.detach().clone().cpu().numpy()

        # original_np and reconstructed_np now have shape (32, 66)
        # which _animate_comparison will correctly reshape to (32, 33, 2)
        ani = self._animate_comparison(fig, ax1, ax2, 
                                       original_np, reconstructed_np,
                                       "Original (Window 0)", "Reconstructed (Window 0)")
        
        plt.show(block=False) # Use block=False to allow second anim
        return ani

    def plot_heatmap(self, feature_mats, titles= None, figsize=(7, 4)):
        """
        Plots multiple heatmaps side by side (horizontal orientation).

        feature_mats: list of tensors (C, T)
        titles: list of titles
        """
        n = len(feature_mats)
        if titles is None:
            titles = [f"Feature {i}" for i in range(n)]
        
        plt.figure(figsize=figsize)
        
        for i, (mat, title) in enumerate(zip(feature_mats, titles), 1):
            plt.subplot(n, 1, i)  # 1 row, n columns
            sns.heatmap(mat.detach().clone().cpu().numpy().T, cmap="coolwarm", cbar=True)  # transpose for horizontal
            plt.title(title)
            plt.xlabel("Feature Channels")
            plt.ylabel("Time")
        
        plt.tight_layout()
        plt.show(block=False)

    def run_cross_reconstruction_test(self, model, window_action_batch, window_skeleton_batch, window_view_batch):
        """
        Tests the model's ability to mix-and-match features.
        Inputs are batches, e.g., shape [BATCH, N_FRAMES, 33, 2]
        """
        print("Running Cross-Reconstruction Test (Displaying first window in batch)...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Test 2: Cross-Reconstruction (a1_s2_v3)")
        
        if not isinstance(window_action_batch, torch.Tensor):
            window_action_batch = torch.tensor(window_action_batch, dtype=torch.float32)
        if not isinstance(window_skeleton_batch, torch.Tensor):
            window_skeleton_batch = torch.tensor(window_skeleton_batch, dtype=torch.float32)
        if not isinstance(window_view_batch, torch.Tensor):
            window_view_batch = torch.tensor(window_view_batch, dtype=torch.float32)
            
        # 2. Flatten the features for all batches: (B, F, 33, 2) -> (B, F, 66)
        w_a = window_action_batch.flatten(start_dim=2).float().to(self.device)
        w_s = window_skeleton_batch.flatten(start_dim=2).float().to(self.device)
        w_v = window_view_batch.flatten(start_dim=2).float().to(self.device)


        with torch.no_grad():
            h_a, _, _ = model.encode(w_a[:8]) # Action from A[0]
            _, h_s, _ = model.encode(w_s[:8]) # Skeleton from B[0]
            _, _, h_v = model.encode(w_v[:8]) # View from C[0]
            
            cross_reconstructed_tensor = model.decode(h_a, h_s, h_v)

        original_action_np = w_a[:8].detach().clone().cpu().numpy() 

        cross_reconstructed_np = cross_reconstructed_tensor.squeeze(0).detach().clone().cpu().numpy()

        ani = self._animate_comparison(fig, ax1, ax2,
                                    original_action_np, cross_reconstructed_np,
                                    "Original Action (a1[0])", "Cross-Reconstructed (a1_s2_v3)")
        
        plt.show(block=False)
        
        return ani
