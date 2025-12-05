import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------
# --- 1. Helper: Spatial Attention (Learns "Connections") ---
# -----------------------------------------------------------------
class SpatialGraphAttention(nn.Module):
    """
    This block looks at the relationships between joints (Spatial)
    BEFORE looking at time. It simulates finding 'angles' and 'bones'
    by learning which joints are relevant to each other.
    """
    def __init__(self, in_features, inner_dim=64):
        super().__init__()
        # We treat the input features (33 joints * 3 coords = 99) 
        # as the "context" we want to understand.
        
        # 1x1 Conv acts as a "Per-Frame" MLP. 
        # It mixes joints together to find geometric poses.
        self.query = nn.Conv1d(in_features, inner_dim, kernel_size=1)
        self.key   = nn.Conv1d(in_features, inner_dim, kernel_size=1)
        self.value = nn.Conv1d(in_features, in_features, kernel_size=1) # Keep dims same
        
        self.amma = nn.Parameter(torch.zeros(1)) # Learnable scalingg
        self.bn = nn.BatchNorm1d(in_features)

    def forward(self, x):
        # x: [Batch, 99, Time]
        
        # Attention across channels (Joints/Coordinates)
        # We want to know: "Does the Hand correlate with the Elbow?"
        Q = self.query(x) # [B, 64, T]
        K = self.key(x)   # [B, 64, T]
        V = self.value(x) # [B, 99, T]
        
        # Calculate correlation between features (Channel Attention)
        # Note: Usually attention is T x T. Here we want Feature x Feature dominance
        # But for 1D Motion, a lightweight "Gating" is often more stable:
        
        attention = F.softmax(torch.matmul(Q.transpose(1, 2), K) / 8.0, dim=-1) 
        # We apply this temporal attention to smooth out "glitches" in time
        # based on spatial consistency.
        
        out = torch.matmul(attention, V.transpose(1, 2)).transpose(1, 2)
        
        out = self.gamma * out + x # Residual connection
        return self.bn(out)

# -----------------------------------------------------------------
# --- 2. Temporal Smoothing Block (Anti-Flicker) ---
# -----------------------------------------------------------------
class SmoothConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        # Reflection Pad mimics natural motion continuity at edges
        # (Instead of Zero padding which shocks the model)
        self.pad = nn.ReflectionPad1d(1) 
        self.conv = nn.Conv1d(in_c, out_c, kernel_size=3, stride=stride, padding=0)
        self.bn = nn.BatchNorm1d(out_c)
        self.act = nn.GELU() # GELU is much smoother than ReLU (No sharp 0 cutoff)

    def forward(self, x):
        x = self.pad(x)
        return self.act(self.bn(self.conv(x)))

# -----------------------------------------------------------------
# --- 3. Geometric Action Encoder ---
# -----------------------------------------------------------------
class GeometricActionEncoder(nn.Module):
    def __init__(self, input_features, latent_dim=128):
        super().__init__()
        
        # --- Stage 1: Spatial Understanding (The "Angles" Proxy) ---
        # Before downsampling time, we mix the 99 features to find "Pose"
        self.spatial_mix = nn.Sequential(
            nn.Conv1d(input_features, 128, kernel_size=1), # Mix Joints
            nn.BatchNorm1d(128),
            nn.GELU(),
            SpatialGraphAttention(128) # Refine relationships
        )
        
        # --- Stage 2: Temporal Smoothing & Downsampling ---
        # Now we process the "Pose Sequence"
        self.t1 = SmoothConvBlock(128, 128, stride=1)       # Smooth
        self.down1 = SmoothConvBlock(128, 128, stride=2)    # T/2
        
        self.t2 = SmoothConvBlock(128, 256, stride=1)       # Smooth
        self.down2 = SmoothConvBlock(256, 256, stride=2)    # T/4
        
        self.t3 = SmoothConvBlock(256, latent_dim, stride=1)
        self.down3 = SmoothConvBlock(latent_dim, latent_dim, stride=2) # T/8

    def forward(self, x):
        # x: [Batch, 99, Frames]
        
        # 1. Learn Geometry (Frame by Frame)
        x = self.spatial_mix(x)
        
        # 2. Learn Dynamics (Time)
        x = self.down1(self.t1(x))
        x = self.down2(self.t2(x))
        x = self.down3(self.t3(x))
        
        return x

# -----------------------------------------------------------------
# --- 4. Robust Static Encoder ---
# -----------------------------------------------------------------
class RobustStaticEncoder(nn.Module):
    def __init__(self, input_features, output_dim=64):
        super().__init__()
        # To get the skeleton, we average the FEATURES, not the time output.
        # We look for features that are constant throughout the clip.
        
        self.net = nn.Sequential(
            nn.Conv1d(input_features, 128, kernel_size=1), # Look at pose
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, padding=1), # Look at local change
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, output_dim, kernel_size=1)
        )
        
    def forward(self, x):
        x = self.net(x) # [B, 64, T]
        
        # Global Average Pooling removes ALL motion, leaving only Structure
        # But we expand it back to [T/8] to match Action Encoder for decoder
        t_out = x.shape[-1] // 8 # Target temporal dim
        
        x = F.adaptive_avg_pool1d(x, 1) # [B, 64, 1]
        
        # Return as [B, 64, 1] (Broadcasting handled in decoder)
        return x 

# -----------------------------------------------------------------
# --- 5. Smooth Decoder (Anti-Jitter) ---
# -----------------------------------------------------------------
class SmoothDecoder(nn.Module):
    def __init__(self, action_dim, static_dim, output_features):
        super().__init__()
        combined = action_dim + static_dim + static_dim
        
        # We use Nearest Neighbor upsampling + Convolution 
        # This is often smoother than Deconvolution (ConvTranspose) which causes checkerboards
        self.layer1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False), # Linear smoothing
            nn.ReflectionPad1d(1),
            nn.Conv1d(combined, 256, 3),
            nn.BatchNorm1d(256),
            nn.GELU()
        )
        
        self.layer2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ReflectionPad1d(1),
            nn.Conv1d(256, 128, 3),
            nn.BatchNorm1d(128),
            nn.GELU()
        )
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='linear', align_corners=False),
            nn.ReflectionPad1d(3), # Large context for final touch
            nn.Conv1d(128, output_features, 7) # No activation at end
        )

    def forward(self, action, skeleton, view):
        # Action: [B, 128, T/8]
        # Skeleton: [B, 64, 1]
        
        T = action.shape[-1]
        
        # Expand static features to match time dimension
        s_exp = skeleton.expand(-1, -1, T)
        v_exp = view.expand(-1, -1, T)
        
        x = torch.cat([action, s_exp, v_exp], dim=1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.final(x)
        return x

# -----------------------------------------------------------------
# --- Main Class ---
# -----------------------------------------------------------------
class FullAutoencoder(nn.Module):
    def __init__(self, input_features=99, num_frames=32, action_dim=128, static_dim=64):
        super().__init__()
        
        # 1. Save Config
        self.num_frames = num_frames
        self.input_features = input_features
        self.action_dim = action_dim
        self.static_dim = static_dim
        
        # 2. Initialize Sub-Modules with the provided dimensions
        # We pass 'action_dim' as 'latent_dim' to the Motion Encoder
        self.E_m = GeometricActionEncoder(input_features, latent_dim=action_dim)
        
        # We pass 'static_dim' as 'output_dim' to the Static Encoders
        self.E_s = RobustStaticEncoder(input_features, output_dim=static_dim)
        self.E_v = RobustStaticEncoder(input_features, output_dim=static_dim)
        
        # Decoder needs both to know input size
        self.D = SmoothDecoder(action_dim=action_dim, static_dim=static_dim, output_features=input_features)
    def encode(self, window):
        x = window.permute(0, 2, 1) # [B, F, T]
        action = self.E_m(x)
        skeleton = self.E_s(x)
        view = self.E_v(x)
        return action, skeleton, view

    def decode(self, action, skeleton, view):
        recon = self.D(action, skeleton, view)
        return recon.permute(0, 2, 1) # [B, T, F]

    def forward(self, window):
        a, s, v = self.encode(window)
        return self.decode(a, s, v)