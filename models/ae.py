import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual Block with Dilation to increase receptive field.
    """
    def __init__(self, channels, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        # Padding for the dilated convolution (conv1)
        padding1 = (kernel_size - 1) * dilation // 2
        
        # Padding for the standard convolution (conv2, dilation=1)
        padding2 = (kernel_size - 1) // 2
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, 
                               stride=1, padding=padding1, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, 
                               stride=1, padding=padding2, dilation=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual  # Skip connection
        return F.leaky_relu(out, 0.2)

class ConvBlock(nn.Module):
    """Standard Conv Block for downsampling"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels) # Added BatchNorm for stability
    
    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

# -----------------------------------------------------------------
# --- 2. IMPROVED Motion Information Encoder (E_M) ---
# -----------------------------------------------------------------
class ActionEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features # Saved for shape checking
        
        # 1. Initial Downsampling / Feature expansion
        self.down1 = ConvBlock(input_features, 64, kernel_size=7, padding=3, stride=2)
        
        # 2. Deep Motion Extraction (Residual + Dilation)
        # Dilation 1, 2, 4 increases receptive field exponentially
        self.res1 = ResidualBlock(64, dilation=1)
        self.down2 = ConvBlock(64, 96, stride=2)
        
        self.res2 = ResidualBlock(96, dilation=2)
        self.down3 = ConvBlock(96, 128, stride=2)
        
        self.res3 = ResidualBlock(128, dilation=4)
        
    def forward(self, x):
        # --- JIT / SHAPE GUARD ---
        # If input is (Batch, Frames, Features) [B, 32, 99] but we expect [B, 99, 32]
        # We auto-permute to fix JIT tracing issues on sub-modules.
        if x.shape[1] != self.input_features and x.shape[2] == self.input_features:
            x = x.permute(0, 2, 1)

        # --- VELOCITY TRICK ---
        # Explicitly calculate velocity (frame_t - frame_t-1)
        # This forces the encoder to look at MOTION, not positions.
        # We pad the first frame to keep shape consistent.
        velocity = torch.zeros_like(x)
        velocity[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        velocity[:, :, 0] = velocity[:, :, 1] # Replicate first frame velocity
        
        # Use velocity as input instead of raw positions
        x = velocity
        
        x = self.down1(x) # T/2
        x = self.res1(x)
        
        x = self.down2(x) # T/4
        x = self.res2(x)
        
        x = self.down3(x) # T/8
        x = self.res3(x)
        
        return x

# -----------------------------------------------------------------
# --- 3. Static Encoder (E_S and E_V) ---
# -----------------------------------------------------------------
class StaticEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        self.input_features = input_features # Saved for shape checking

        self.conv1 = ConvBlock(input_features, 32, kernel_size=7, padding=3, stride=2)
        self.conv2 = ConvBlock(32, 48, stride=2)
        self.conv3 = ConvBlock(48, 64, stride=2)

    def forward(self, x):
        # --- JIT / SHAPE GUARD ---
        if x.shape[1] != self.input_features and x.shape[2] == self.input_features:
            x = x.permute(0, 2, 1)

        # Static encoder sees raw POSITIONS, not velocity
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

# -----------------------------------------------------------------
# --- 4. Decoder (D) ---
# -----------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, action_channels, static_channels, output_features, num_frames):
        super().__init__()
        self.combined_dim = action_channels + static_channels + static_channels 

        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False) # Linear is smoother for motion
        self.conv1 = nn.Conv1d(self.combined_dim, 128, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(128)

        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv2 = nn.Conv1d(128, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(64)

        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv_final = nn.Conv1d(64, output_features, kernel_size=7, stride=1, padding=3)

    def forward(self, action_feat, skeleton_feat, view_feat):
        # Combine
        x = torch.cat([action_feat, skeleton_feat, view_feat], dim=1) 
        
        x = self.up1(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)

        x = self.up2(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)

        x = self.up3(x)
        x = self.conv_final(x) # No activation at output (regression)

        return x

# -----------------------------------------------------------------
# --- Main Wrapper ---
# -----------------------------------------------------------------
class FullAutoencoder(nn.Module):
    def __init__(self, input_features, num_frames, action_dim=128, static_dim=64):
        super().__init__()
        
        if num_frames % 8 != 0:
            raise ValueError(f"N_FRAMES ({num_frames}) must be divisible by 8.")
            
        self.input_features = input_features
        self.num_frames = num_frames
        
        self.E_m = ActionEncoder(input_features) # Input features -> Velocity internal calculation
        self.E_s = StaticEncoder(input_features)
        self.E_v = StaticEncoder(input_features)
        
        self.D = Decoder(action_dim, static_dim, input_features, num_frames)

    def _format_input(self, window):
        if window.dim() != 3 or window.shape[1] != self.num_frames:
             raise ValueError(f"Expected input shape (batch, {self.num_frames}, ...)")
        return window.permute(0, 2, 1)

    def encode(self, window):
        window_t = self._format_input(window)
        
        # E_m calculates velocity internally now
        action_feat = self.E_m(window_t)
        
        # E_s and E_v see raw positions
        skeleton_feat = self.E_s(window_t)
        view_feat = self.E_v(window_t)
        
        return action_feat, skeleton_feat, view_feat

    def decode(self, action_feat, skeleton_feat, view_feat):
        reconstructed_window_t = self.D(action_feat, skeleton_feat, view_feat)
        return reconstructed_window_t.permute(0, 2, 1)

    def forward(self, window):
        action_feat, skeleton_feat, view_feat = self.encode(window)
        reconstructed_window = self.decode(action_feat, skeleton_feat, view_feat)
        return reconstructed_window