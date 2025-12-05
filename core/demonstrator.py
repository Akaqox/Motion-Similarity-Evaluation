import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mediapipe as mp
import os
import seaborn as sns


class Demonstrator:
    """
    A class to demonstrate the capabilities of a trained
    FullAutoencoder model.
    """
    def __init__(self, config):
        self.ae_cfg = config.get("ae")
        self.ws_cfg = config.get("window")

        self.device = torch.device(self.ae_cfg["device"])
        self.n_frames = self.ws_cfg["size"]
        self.n_features = self.ae_cfg["N_FEATURES"]
        self.POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

    def _animate_comparison(self, fig, ax1, ax2, window_batch1, window_batch2, title1, title2):
        """
        Fixed animation with INDEPENDENT axis scaling.
        This prevents one skeleton from appearing as a dot if their scales differ.
        """
        # Reduce Figure DPI for performance
        fig.set_dpi(72)

        # 0. Input Safety: Ensure inputs are 3D (Batch, Frames, Features)
        if window_batch1.ndim == 2: window_batch1 = window_batch1[np.newaxis, ...]
        if window_batch2.ndim == 2: window_batch2 = window_batch2[np.newaxis, ...]

        batch_size, n_frames, n_feats = window_batch1.shape

        # 1. Unified Projection Logic
        def project_to_2d(batch):
            if batch.shape[-1] == 99: # 3D data
                return batch.reshape(batch_size, n_frames, 33, 3)[..., :2]
            elif batch.shape[-1] == 66: # 2D data
                return batch.reshape(batch_size, n_frames, 33, 2)
            else:
                raise ValueError(f"Unexpected feature size: {batch.shape[-1]}")

        frames1 = project_to_2d(window_batch1)
        frames2 = project_to_2d(window_batch2)
        
        # 2. Setup INDEPENDENT Axis Limits
        # We calculate limits separately for Left (ax1) and Right (ax2)
        def get_limits(data):
            # Filter out NaNs just in case
            valid = data[~np.isnan(data).any(axis=-1)]
            if len(valid) > 0:
                return (valid[..., 0].min(), valid[..., 0].max(),
                        valid[..., 1].min(), valid[..., 1].max())
            return -1, 1, -1, 1

        lim1 = get_limits(frames1)
        lim2 = get_limits(frames2)
        
        padding = 0.2

        # Apply limits separately
        def apply_limits(ax, lims):
            x_min, x_max, y_min, y_max = lims
            # If the model outputs all zeros/constants, add padding so plot doesn't crash
            if x_min == x_max: x_min -= 1; x_max += 1
            if y_min == y_max: y_min -= 1; y_max += 1
            
            ax.set_xlim(x_min - padding, x_max + padding)
            ax.set_ylim(y_min - padding, y_max + padding)
            ax.invert_yaxis()
            ax.grid(False)
            ax.axis('off') # Clean look
        
        apply_limits(ax1, lim1)
        apply_limits(ax2, lim2)

        # 3. Initialize Plot Objects (Draw once)
        def init_skeleton_artists(ax, color):
            lines = []
            for _ in self.POSE_CONNECTIONS:
                line, = ax.plot([], [], f'{color}-', lw=2, alpha=0.7)
                lines.append(line)
            scatter = ax.scatter([], [], s=15, c=color, zorder=3)
            return lines, scatter

        lines1, scatter1 = init_skeleton_artists(ax1, 'b')
        lines2, scatter2 = init_skeleton_artists(ax2, 'r')
        
        txt1 = ax1.text(0.5, 1.05, title1, transform=ax1.transAxes, ha='center')
        txt2 = ax2.text(0.5, 1.05, title2, transform=ax2.transAxes, ha='center')

        total_animation_frames = batch_size * n_frames

        # 4. Update Loop (Blitting optimized)
        def update(i):
            batch_idx = i // n_frames
            frame_idx = i % n_frames
            
            def update_skeleton(data, lines, scatter, txt, base_title):
                scatter.set_offsets(data)
                
                for line, (start, end) in zip(lines, self.POSE_CONNECTIONS):
                    if start < 33 and end < 33:
                        x_pts = [data[start, 0], data[end, 0]]
                        y_pts = [data[start, 1], data[end, 1]]
                        line.set_data(x_pts, y_pts)
                
                txt.set_text(f"{base_title}\nWindow {batch_idx}, Frame {frame_idx}")
                return lines + [scatter, txt]

            artists1 = update_skeleton(frames1[batch_idx, frame_idx], lines1, scatter1, txt1, title1)
            artists2 = update_skeleton(frames2[batch_idx, frame_idx], lines2, scatter2, txt2, title2)
            
            return artists1 + artists2

        ani = FuncAnimation(fig, update, frames=total_animation_frames, interval=200, blit=True)
        return ani

    def run_reconstruction_test(self, model, original_window_batch):
        print("Running Reconstruction Test...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        plt.suptitle("Test 1: Reconstruction")

        if not isinstance(original_window_batch, torch.Tensor):
            original_window_batch = torch.tensor(original_window_batch, dtype=torch.float32)

        window_batch_tensor = original_window_batch.flatten(start_dim=2).float().to(self.device)
        
        with torch.no_grad():
            reconstructed_batch_tensor = model(window_batch_tensor)
            
        original_np = window_batch_tensor.detach().cpu().numpy()
        reconstructed_np = reconstructed_batch_tensor.detach().cpu().numpy()

        ani = self._animate_comparison(fig, ax1, ax2, 
                                       original_np, reconstructed_np,
                                       "Original", "Reconstructed")
        plt.show(block=False)
        return ani

    def run_cross_reconstruction_test(self, model, window_action_batch, window_skeleton_batch, window_view_batch):
        print("Running Cross-Reconstruction Test...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.suptitle("Test 2: Cross-Reconstruction")
        
        # Convert inputs
        def to_tensor(x):
            t = torch.tensor(x, dtype=torch.float32) if not isinstance(x, torch.Tensor) else x
            # Fix 99->66 mismatch if model requires it
            if t.shape[-1] == 3 and model.input_features == 66:
                t = t[..., :2]
            return t.flatten(start_dim=2).float().to(self.device)

        w_a = to_tensor(window_action_batch)
        w_s = to_tensor(window_skeleton_batch)
        w_v = to_tensor(window_view_batch)

        # Slice a batch of 8 (or less if data is smaller)
        batch_slice = 8
        w_a = w_a[:batch_slice]
        w_s = w_s[:batch_slice]
        w_v = w_v[:batch_slice]

        with torch.no_grad():
            h_a, _, _ = model.encode(w_a) # Action from A
            _, h_s, _ = model.encode(w_s) # Skeleton from B
            _, _, h_v = model.encode(w_v) # View from C
            
            cross_reconstructed_tensor = model.decode(h_a, h_s, h_v)

        original_action_np = w_a.detach().cpu().numpy() 
        
        # FIX: Remove .squeeze(0) because we are processing a batch, not a single item
        cross_reconstructed_np = cross_reconstructed_tensor.detach().cpu().numpy()

        ani = self._animate_comparison(fig, ax1, ax2,
                                    original_action_np, cross_reconstructed_np,
                                    "Original Action Source", "Cross-Reconstructed")
        
        plt.show(block=False)
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
