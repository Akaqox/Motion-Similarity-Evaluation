import torch
import os
import numpy as np
from core.demonstrator import Demonstrator as DS
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from data.ws import Window_Slider as WS
from models.ae import FullAutoencoder
from scipy.spatial.distance import cosine, euclidean, sqeuclidean, cdist
from fastdtw import fastdtw
import matplotlib.collections as mc

class Inference():

    def __init__(self, cfg, m_path = ""):
        self.ws_cfg = cfg.get("window")
        self.ae_cfg = cfg.get("ae")
        self.cfg = cfg

        self.n_frames = self.ws_cfg["size"]
        self.stride = self.ws_cfg["stride"]
        self.matching = self.ws_cfg["matching"]
        self.n_features = self.ae_cfg["N_FEATURES"]

        self.device = torch.device(self.ae_cfg["device"])
        print(m_path)

        if m_path != "" and cfg.get("eval", "custom"):
            self.m_model = self._load_model(m_path)
            model_path = f'{self.ae_cfg["full_ae_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
        else:
            m_path = f'{self.ae_cfg["e_m_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
            model_path = f'{self.ae_cfg["full_ae_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
            
            self.model = self._load_model(model_path)
            self.m_model = self._load_model(m_path)
            print(f"Loading full model from {model_path}...")
        



    def _load_model(self, model_path):
        # 1. Check file existence first to avoid try/catch overhead for missing files
        try:
            # Attempt 1: Standard state_dict load (updates existing model)
            model = torch.load(model_path, weights_only=False, map_location=self.device)

        except (RuntimeError, Exception) as e:
            print(f"Standard load failed ({e}). Attempting JIT fallback...")
            
            # Attempt 2: Fallback to JIT load (creates NEW model object)
            try:
                model = torch.jit.load(model_path, map_location=self.device)
                print("Success: Loaded model via JIT.")
            except Exception as jit_e:
                raise RuntimeError(f"FATAL: Failed to load as state_dict OR JIT.\nOriginal: {e}\nJIT: {jit_e}")

        # Unified configuration
        model.to(self.device)
        model.eval()
        
        return model

    def _load_windows(self, path):
        if isinstance(path, str):
            seq = np.load(path)
        else:
            seq = path
        slider = WS(self.ws_cfg["size"], self.ws_cfg["stride"])
        windows = slider.slide(seq)
        if self.n_features == 66:
            windows = windows[..., :2] 

        w_tensor = torch.tensor(windows)
        return w_tensor

    def get_encoder_fe(self,  window):
        """
        Returns raw encoder features separately:
        - action_features:   (C1, T)
        - skeleton_features: (C2, T)
        - view_features:     (C3, T)
        """
        with torch.no_grad():
            w = torch.tensor(window, dtype=torch.float32).to(self.device)
            if len(w.size()) == 3:
                w = w.unsqueeze(0).flatten(start_dim=2)  # B=1
            else:
                w = w.flatten(start_dim=2)

            print(f"Shape before transpose check: {w.shape}")
            if w.shape[-1] == self.n_features:
                w = w.permute(0, 2, 1)

            action_feat= self.m_model(w)
            
        return (
            action_feat.squeeze(0)
        )
        
    def rc_anchor(self, ds, anchor, title):

        ani = ds.run_reconstruction_test(self.model, anchor)
        # 2. Pick a single window
        w = anchor[len(anchor)//2]

        with torch.no_grad():
            w = w.to(self.device).float()
            if len(w.size()) == 3:
                w = w.unsqueeze(0).flatten(start_dim=2)  # B=1
            else:
                w = w.flatten(start_dim=2)

            action_f, skeleton_f, view_f = self.model.encode(w)

        
        feature_mats = [action_f.squeeze(0), 
                        skeleton_f.squeeze(0), 
                        view_f.squeeze(0)]
        print(action_f.size())

        titles = [
            f"{title} - Action Features (Temporal)",
            f"{title} - Skeleton Features (Static)",
            f"{title} - View Features (Static)"
        ]

        ds.plot_heatmap(feature_mats, titles)
        return ani

    def demonstrate(self, anchor_path1=None, anchor_path2=None, anchor_path3=None):
        # Load anchors safely
        anchor1 = self._load_windows(anchor_path1)
        anchor2 = self._load_windows(anchor_path2)
        anchor3 = self._load_windows(anchor_path3)

        ds = DS(self.cfg)

        ani1 = self.rc_anchor(ds, anchor1, "Anchor 1")
        ani2 = self.rc_anchor(ds, anchor2, "Anchor 2")
        plt.show()
        ani3 = self.rc_anchor(ds, anchor3, "Anchor 3")

        if anchor1 is not None and anchor2 is not None and anchor3 is not None:
            print("\n=== Cross Reconstruction ===")
            ani4 = ds.run_cross_reconstruction_test(self.model, anchor1, anchor2, anchor3)
            plt.show()
        else:
            ani4 = None

        return ani1, ani2, ani3, ani4

    def visualize_dtw_alignment(self, f1, f2, path):
        """
        f1, f2: The feature arrays used in DTW (N x D)
        path: The list of tuples [(i, j), ...] returned by fastdtw
        """
        # --- Visual 1: The Connection Plot ---
        plt.figure(figsize=(12, 4))

        # We use the norm of the features just to have a 1D signal to plot
        sig1 = np.linalg.norm(f1, axis=1)
        # Shift sig2 down so we can see the lines clearly
        sig2 = np.linalg.norm(f2, axis=1) - (np.max(sig1) * 2)

        plt.plot(sig1, label='Seq 1 (Feature Norm)', color='blue')
        plt.plot(sig2, label='Seq 2 (Feature Norm)', color='orange')

        # Create lines connecting matched indices
        lines = []
        for p in path:
            # p[0] is index in sig1, p[1] is index in sig2
            # (x, y) coordinates: (index, signal_value)
            lines.append([(p[0], sig1[p[0]]), (p[1], sig2[p[1]])])

        lc = mc.LineCollection(lines, colors='grey', linewidths=0.5, alpha=0.5)
        plt.gca().add_collection(lc)

        plt.title("DTW Alignment: Lines show which Patch matches which")
        plt.legend()
        plt.tight_layout()
        plt.show(block=False)

        # --- Visual 2: The Cost Matrix & Path ---
        # Calculate the full distance matrix to see the 'landscape'
        dist_mat = cdist(f1, f2, metric='cosine')

        plt.figure(figsize=(8, 8))
        plt.imshow(dist_mat, origin='lower', cmap='viridis', aspect='auto')
        plt.colorbar(label='Cosine Distance')

        # Unzip path for plotting
        path_i, path_j = zip(*path)
        plt.plot(path_j, path_i, 'w-', linewidth=2, label='Optimal Path')

        plt.xlabel("Sequence 2 Index")
        plt.ylabel("Sequence 1 Index")
        plt.title("Alignment Path over Cost Matrix")
        plt.legend()
        plt.show()

        # --- Text Output: First 10 matches ---
        print("Sample Matches (Seq1_Idx -> Seq2_Idx):")
        for i, j in path[:10]:
            print(f"Patch {i} -> Patch {j}")

    def dtw_similarity(self, seq1, seq2, vis=False):
        seq1 = self._load_windows(seq1)
        seq2 = self._load_windows(seq2)

        f1 = self.get_encoder_fe(seq1).detach().cpu().numpy()
        f2 = self.get_encoder_fe(seq2).detach().cpu().numpy()

        f1 = f1.reshape(f1.shape[0], -1)
        f2 = f2.reshape(f2.shape[0], -1)
        
        # Center the data
        f1 = f1 - np.mean(f1, axis=0)
        f2 = f2 - np.mean(f2, axis=0)

        # Basic DTW with Cosine
        _, path = fastdtw(f1, f2, dist=cosine, radius=1)
        
        path = np.array(path)
        aligned_f1 = f1[path[:, 0]]
        aligned_f2 = f2[path[:, 1]]

        dots = np.sum(aligned_f1 * aligned_f2, axis=1)
        norms = np.linalg.norm(aligned_f1, axis=1) * np.linalg.norm(aligned_f2, axis=1)
        
        # Raw Similarity Scores
        S = np.divide(dots, norms, out=np.zeros_like(dots), where=norms!=0)


        start_trim_thresh = 0.5
        
        # Skoru 0.1'den büyük olan ilk ve son indexi bul
        valid_indices = np.where(S > start_trim_thresh)[0]

        if len(valid_indices) > 5:  # En az 5 karelik anlamlı hareket varsa kesme yap
            start_idx = valid_indices[0]
            end_idx = valid_indices[-1]
            
            # Sadece anlamlı aralığı al
            S_trimmed = S[start_idx : end_idx + 1]
            

            if len(S_trimmed) < len(S) * 0.2:
                S_active = S 
            else:
                S_active = S_trimmed
        else:

            S_active = S

        th_h = 0.3
        th_l = 0.6
        penalty_severity_high = 25.0 
        penalty_severity_low = 5.0

        gap_h = th_h - S_active
        gap_l = th_l - S_active

        conditions = [
            gap_h > 0,  # S < 0.3
            gap_l > 0   # S < 0.6
        ]

        choices = [
            S_active - (gap_h * penalty_severity_high),
            S_active - (gap_l * penalty_severity_low)
        ]

        S_weighted = np.select(conditions, choices, default=S_active)
        
        S_avg = np.mean(S_weighted)

        # Clip final result to 0-1
        final_score = float(np.clip(S_avg, 0.0, 1.0))

        print(f"Original Len: {len(S)}, Active Len: {len(S_active)}")
        print(f"Final Score: {final_score}")
        
        if vis:
            self.visualize_dtw_alignment(f1, f2, path)

        return final_score, S_avg, S