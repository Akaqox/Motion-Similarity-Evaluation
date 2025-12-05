import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConvBlock(nn.Module):
    """
    A simple block: 1D Convolution -> Leaky ReLU
    (Based on the repeating units in Fig. 5)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):
        return F.leaky_relu(self.conv(x), 0.2)

# -----------------------------------------------------------------
# --- 2. Motion Information Encoder (E_M) ---
# -----------------------------------------------------------------
class ActionEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        
        self.conv1 = ConvBlock(input_features, 64, kernel_size=7, padding=3)
        self.conv2 = ConvBlock(64, 96)
        self.conv3 = ConvBlock(96, 128) 
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) 
        return x

# -----------------------------------------------------------------
# --- 3. Static Encoder (E_S and E_V) ---
# -----------------------------------------------------------------
class StaticEncoder(nn.Module):
    def __init__(self, input_features):
        super().__init__()
        # Mirror the action encoder's downsampling so both produce T/8
        # Channels chosen according to your diagram: 32 -> 48 -> 64
        self.conv1 = ConvBlock(input_features, 32, kernel_size=7, padding=3, stride=2)  # /2
        self.conv2 = ConvBlock(32, 48, stride=2)  # /4
        self.conv3 = ConvBlock(48, 64, stride=2)  # /8
        # DO NOT global-pool here — keep temporal dimension
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # output: [B, 64, T/8]
        return x

# -----------------------------------------------------------------
# --- 4. Decoder (D) ---
# -----------------------------------------------------------------
class Decoder(nn.Module):
    def __init__(self, action_channels, static_channels, output_features, num_frames):
        super().__init__()
        self.num_frames = num_frames

        # compute combined channels from actual encoder channel sizes:
        self.combined_dim = action_channels + static_channels + static_channels  # e.g. 128 + 64 + 64 = 256

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop1 = nn.Dropout(0.2)
        self.conv1 = ConvBlock(self.combined_dim, 128, kernel_size=3, stride=1, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop2 = nn.Dropout(0.2)
        self.conv2 = ConvBlock(128, 64, kernel_size=3, stride=1, padding=1)

        self.up3 = nn.Upsample(scale_factor=2, mode='nearest')
        self.drop3 = nn.Dropout(0.2)
        self.conv_final = nn.Conv1d(64, output_features, kernel_size=7, stride=1, padding=3)

    def forward(self, action_feat, skeleton_feat, view_feat):
        x = torch.cat([action_feat, skeleton_feat, view_feat], dim=1)  # [B, combined_dim, T/8]
        # Upsample and convs — keep upsampling inside decoder
        x = self.up1(x)
        x = self.conv1(self.drop1(x))

        x = self.up2(x)
        x = self.conv2(self.drop2(x))

        x = self.up3(x)
        x = self.conv_final(self.drop3(x))

        return x

#The Full Autoencoder System 
class FullAutoencoder(nn.Module):
    """
    Holds all three encoders and the decoder.
    
    **This architecture requires `n_frames` to be
    divisible by 8 (due to the 3 pooling/upsampling layers).
    Please change `window_ size` to 16 or 32 in your config.
    """
    def __init__(self, input_features, num_frames, action_dim, static_dim):
        super().__init__()
        
        if num_frames % 8 != 0:
            raise ValueError(f"N_FRAMES ({num_frames}) must be divisible by 8 for this model architecture.")
            
        self.input_features = input_features
        self.num_frames = num_frames
        
        # Instantiate sub-modules with the provided dimensions
        self.E_m = ActionEncoder(input_features)
        self.E_s = StaticEncoder(input_features)
        self.E_v = StaticEncoder(input_features)
        self.D = Decoder(action_dim, static_dim, input_features, num_frames)

    def _format_input(self, window):
        """
        Takes input (batch, frames, features)
        Returns (batch, features, frames)
        """
        if window.dim() != 3 or window.shape[1] != self.num_frames or window.shape[2] != self.input_features:
             raise ValueError(f"Expected input shape (batch, {self.num_frames}, {self.input_features}), "
                              f"but got {window.shape}")
        
        return window.permute(0, 2, 1)

    def encode(self, window):
        """
        Separates a window into its three feature vectors.
        Input: window (batch, n_frames, n_features)
        """
        window_t = self._format_input(window) # (batch, n_features, n_frames)
        
        action_feat = self.E_m(window_t)
        skeleton_feat = self.E_s(window_t)
        view_feat = self.E_v(window_t)
        
        return action_feat, skeleton_feat, view_feat

    def decode(self, action_feat, skeleton_feat, view_feat):
        """
        Reconstructs a window from three feature vectors.
        Output: (batch, n_frames, n_features)
        """
        reconstructed_window_t = self.D(action_feat, skeleton_feat, view_feat)
        return reconstructed_window_t.permute(0, 2, 1)

    def forward(self, window):
        """
        A standard pass: Encode and reconstruct the same window.
        Input: window (batch, n_frames, n_features)
        Output: reconstructed_window (batch, n_frames, n_features)
        """
        action_feat, skeleton_feat, view_feat = self.encode(window)
        reconstructed_window = self.decode(action_feat, skeleton_feat, view_feat)
        return reconstructed_window

