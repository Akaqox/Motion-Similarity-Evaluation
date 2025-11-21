import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation # <-- IMPORT THIS
import mediapipe as mp # For POSE_CONNECTIONS
import warnings
import os

# --- Import your custom modules ---
from models.ae import FullAutoencoder

# --- CONFIGURE THIS ---
# Make sure this matches your 'main.py' config
conf = {
    "ae": {
        "N_FRAMES": 32,   # <-- MUST BE <= 48 and divisible by 8
        "N_FEATURES": 66, # (33 joints * 2D)
        "ACTION_DIM": 128,
        "STATIC_DIM": 64,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
    }
}
# --- End of Config ---

# --- Helper function for plotting ---
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS

def plot_skeleton(ax, skeleton, color, label):
    """Helper function to plot a single 2D skeleton."""
    # Ensure skeleton is (33, 2)
    if skeleton.shape != (33, 2):
        # Reshape if it's flat (66,)
        if skeleton.shape == (66,):
            skeleton = skeleton.reshape(33, 2)
        else:
            raise ValueError(f"Expected skeleton shape (33, 2), but got {skeleton.shape}")
        
    x_coords = skeleton[:, 0]
    y_coords = skeleton[:, 1]
    
    # Plot Bones
    for (start_idx, end_idx) in POSE_CONNECTIONS:
        if start_idx < 33 and end_idx < 33:
            x_line = [x_coords[start_idx], x_coords[end_idx]]
            y_line = [y_coords[start_idx], y_coords[end_idx]]
            line_label = label if start_idx == 0 and end_idx == 1 else None
            ax.plot(x_line, y_line, f'{color}-', alpha=0.7, label=line_label)
    
    # Plot Joints
    ax.scatter(x_coords, y_coords, s=15, c=color, zorder=3)
    
    # Add keypoint numbers
    for i in range(len(x_coords)):
        ax.text(x_coords[i] + 0.01, y_coords[i], str(i), fontsize=8)

def test_reconstruction(config, model_path, test_file_path):
    
    ae_cfg = config["ae"]
    device = torch.device(ae_cfg["DEVICE"])
    N_FRAMES = ae_cfg["N_FRAMES"]
    N_FEATURES = ae_cfg["N_FEATURES"]

    # 1. Load the FULL Model
    print(f"Loading full model from {model_path}...")
    model = FullAutoencoder(
        input_features=N_FEATURES,
        num_frames=N_FRAMES,
        action_dim=ae_cfg["ACTION_DIM"],
        static_dim=ae_cfg["STATIC_DIM"]
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval() # Set model to evaluation mode
    print("Model loaded successfully.")

    # 2. Load the Test File
    print(f"Loading test file from {test_file_path}...")
    try:
        full_sequence = np.load(test_file_path) # Shape: (T, 33, 2) OR (T, 66)
    except FileNotFoundError:
        print(f"Error: Test file not found at {test_file_path}")
        return
        
    print(f"Loaded sequence shape: {full_sequence.shape}")

    # 3. Check file length
    if len(full_sequence) < N_FRAMES:
        print(f"Error: Test file has {len(full_sequence)} frames, "
              f"but model requires {N_FRAMES} frames.")
        return
        
    # 4. Get a window from the middle of the sequence
    start_frame = (len(full_sequence) - N_FRAMES) // 2
    window = full_sequence[start_frame : start_frame + N_FRAMES]
    
    # 5. *** FIX: Flatten the window ***
    # Reshape from (16, 33, 2) -> (16, 66)
    try:
        window_flat = window.reshape(N_FRAMES, N_FEATURES)
    except ValueError:
        print(f"Error: Could not reshape window from {window.shape} to ({N_FRAMES}, {N_FEATURES}).")
        print("Make sure your N_FEATURES config is correct (e.g., 66 for 2D, 99 for 3D).")
        return

    # 6. Prepare Tensor
    # Add a batch dimension [1, 16, 66]
    window_tensor = torch.tensor(window_flat, dtype=torch.float32).unsqueeze(0).to(device)

    # 7. Get Reconstruction
    with torch.no_grad():
        reconstructed_window_tensor = model(window_tensor)

    # 8. Convert back to NumPy for plotting
    # Squeeze to remove batch dim: [16, 66]
    original_np = window_tensor.squeeze(0).cpu().numpy()
    reconstructed_np = reconstructed_window_tensor.squeeze(0).cpu().numpy()

    # --- 9. ANIMATE THE RESULTS ---
    print("Preparing animation...")

    # Reshape from [16, 66] -> [16, 33, 2]
    original_frames = original_np.reshape(N_FRAMES, 33, 2)
    reconstructed_frames = reconstructed_np.reshape(N_FRAMES, 33, 2)

    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.suptitle("Model Reconstruction Test")

    # Find global axis limits so the plot doesn't "jiggle"
    all_data = np.concatenate((original_frames, reconstructed_frames))
    x_min, x_max = all_data[..., 0].min(), all_data[..., 0].max()
    y_min, y_max = all_data[..., 1].min(), all_data[..., 1].max()
    padding = 0.2 # Add some padding

    def set_axes_limits(ax):
        """Helper to set consistent plot limits and labels."""
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True)
        ax.axis('equal')
        ax.invert_yaxis()

    # This function is called for each frame of the animation
    def update(frame_index):
        ax1.clear()
        ax2.clear()

        # Get the skeleton for the current frame
        original_frame = original_frames[frame_index]
        reconstructed_frame = reconstructed_frames[frame_index]
        
        # Plot Original (left)
        plot_skeleton(ax1, original_frame, 'b', 'Original')
        ax1.set_title(f"Original (Frame {start_frame + frame_index})")
        set_axes_limits(ax1)
        ax1.legend(loc='upper left')

        # Plot Reconstructed (right)
        plot_skeleton(ax2, reconstructed_frame, 'r', 'Reconstructed')
        ax2.set_title(f"Reconstructed (Frame {start_frame + frame_index})")
        set_axes_limits(ax2)
        ax2.legend(loc='upper left')

    # Create the animation
    # interval=60ms is approx 16.6 FPS
    ani = FuncAnimation(fig, update, frames=N_FRAMES, interval=60, blit=False)

    # Show the animation
    plt.show()

if __name__ == "__main__":
    
    # --- CONFIGURE THESE ---
    MODEL_PATH = "full_autoencoder_final.pth"
    
    # IMPORTANT: Use a file from your VALIDATION set!
    TEST_FILE_PATH = "data/ready_train/a21_s7_t3_v2.npy" # This is your file
    # -----------------------

    if not os.path.exists(TEST_FILE_PATH):
        print(f"Error: Test file not found at {TEST_FILE_PATH}")
    elif not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Did you re-train and save the full model?")
    else:
        test_reconstruction(conf, MODEL_PATH, TEST_FILE_PATH)