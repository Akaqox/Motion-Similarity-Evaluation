import torch
import numpy as np
from core.demonstrator import Demonstrator as DS
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from data.ws import Window_Slider as WS
from models.ae import FullAutoencoder
from scipy.spatial.distance import cosine, euclidean, sqeuclidean
from fastdtw import fastdtw

class Inference():

    def __init__(self, cfg):
        self.ws_cfg = cfg.get("window")
        self.ae_cfg = cfg.get("ae")
        self.cfg = cfg

        self.n_frames = self.ws_cfg["size"]
        self.stride = self.ws_cfg["stride"]
        self.matching = self.ws_cfg["matching"]
        self.n_features = self.ae_cfg["N_FEATURES"]
        
        m_path = f'{self.ae_cfg["e_m_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
        model_path = f'{self.ae_cfg["full_ae_path"]}_{self.matching}_{self.n_features}_{self.n_frames}_{self.stride}.pth'
        self.device = torch.device(self.ae_cfg["device"])

        print(f"Loading full model from {model_path}...")
        
        # Check if the model class is available
        if 'FullAutoencoder' not in globals():
            print("FATAL: FullAutoencoder class is not defined.")
            print("Please ensure 'models.ae' is imported correctly.")
            exit()
        
        self.model = FullAutoencoder(
            input_features=self.n_features,
            num_frames=self.n_frames,
            action_dim=self.ae_cfg["ACTION_DIM"],
            static_dim=self.ae_cfg["STATIC_DIM"]
        )
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        except FileNotFoundError:
            print(f"FATAL: Model file not found at {model_path}")
            print("Please run train.py to generate the model file.")
            exit()
        except RuntimeError as e:
            print(f"FATAL: Error loading model state_dict.")
            print("This usually means your 'conf' does not match the saved model.")
            print(f"Error details: {e}")
            exit()
             
        self.m_model = torch.load(m_path, map_location=self.device, weights_only=False)
        self.model.to(self.device)
        
        self.m_model.eval()
        self.model.eval() 
        print("Model loaded successfully.")

    def _load_windows(self, path):
        seq = np.load(path)
        slider = WS(self.ws_cfg["size"], self.ws_cfg["stride"])
        windows = slider.slide(seq)
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
            w = w.transpose(1, 2)


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
        anchor1 = self._load_windows(anchor_path1) if isinstance(anchor_path1, str) else None
        anchor2 = self._load_windows(anchor_path2) if isinstance(anchor_path2, str) else None
        anchor3 = self._load_windows(anchor_path3) if isinstance(anchor_path3, str) else None

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
                
    def cosine_similarity(self, v1, v2):
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return np.dot(v1, v2) / denom


    def dtw_similarity(self, seq1, seq2):

        seq1 = self._load_windows(seq1)
        seq2 = self._load_windows(seq2)

        f1 = self.get_encoder_fe(seq1).detach().clone().cpu().numpy()
        f2 = self.get_encoder_fe(seq2).detach().clone().cpu().numpy()

        f1 = f1.reshape(f1.shape[0], -1)
        f2 = f2.reshape(f2.shape[0], -1)

        f1 = (f1 - f1.mean(axis=0)) / (f1.std(axis=0) + 1e-6)
        f2 = (f2 - f2.mean(axis=0)) / (f2.std(axis=0) + 1e-6)
        _, path = fastdtw(f1, f2, dist=lambda a, b: np.linalg.norm(a - b))

        S = []

        for (i, j) in path:
            s_ij = self.cosine_similarity(f1[i], f2[j])
            S.append(s_ij)

        S = np.array(S)
        p = -3.0

        S_avg = (np.mean(np.clip(S, 1e-6, None) ** p)) ** (1.0 / p)
        S_avg = np.clip(S_avg, 0.0, 1.0)
        score = 1.0 / (S_avg + 1.0)
        similarity_0_3 = 6 * (1 - score)
        return float(np.round(similarity_0_3, 1)), S_avg, S