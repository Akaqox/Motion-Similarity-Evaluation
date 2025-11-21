import cv2
import mediapipe as mp
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- New Function: Draw 3D Pose ---
def draw_3d_pose(ax, landmarks, pose_connections):
    """
    Draws a 3D skeleton using Matplotlib.
    Args:
        ax: Matplotlib 3D axes
        landmarks: np.array of shape (num_joints, 3)
        pose_connections: list of (start, end) tuples
    """
    ax.cla()  # clear previous frame
    ax.set_title("3D Pose")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Set axis limits and viewpoint
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=45, azim=80)

    # Flip X for mirrored view
    xs, ys, zs = -landmarks[:, 0], landmarks[:, 1], landmarks[:, 2]

    # Draw bones
    for start, end in pose_connections:
        ax.plot([xs[start], xs[end]],
                [ys[start], ys[end]],
                [zs[start], zs[end]], 'b-', linewidth=2)

    # Draw joints
    ax.scatter(xs, ys, zs, c='r', s=25)

    plt.pause(0.001)