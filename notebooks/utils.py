import plotly.express as px
import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.animation import FuncAnimation


def get_simulation_by_id(h5_path, sim_id):
    """
    Load a specific simulation 'id' from a HDF5 file.
    Returns: torch.Tensor [Nt, C, H, W]
    """
    # Convert ID to string if it's stored as a key
    sid = str(sim_id).zfill(4)

    with h5py.File(h5_path, "r") as f:
        if sid not in f:
            raise KeyError(f"Simulation ID {sid} not found in {h5_path}")

        # Access the data: [Nt, C, H, W]
        data = f[sid]["data"][:]

        # Convert to torch tensor
        data_tensor = torch.from_numpy(data).float().permute(0, 3, 1, 2)

    return data_tensor


def plot_interactive(pred_list, gt_list, channel_idx=0, vmin=None, vmax=None):
    # Ensure same length
    max_steps = min(pred_list.shape[0], gt_list.shape[0])
    p = pred_list[:max_steps, channel_idx]
    g = gt_list[:max_steps, channel_idx]

    # Precompute Absolute Error for visualization
    error = torch.abs(g - p)

    # Precompute RMSE per step: sqrt(mean((g-p)^2))
    # Shape: (max_steps,)
    rmse_per_step = torch.sqrt(torch.mean((g - p) ** 2, dim=[-2, -1])).cpu().numpy()

    # Stack [Time, Type, H, W]
    combined = torch.stack([g, p, error], dim=1).cpu().numpy()

    vrange = [vmin, vmax] if (vmin is not None and vmax is not None) else None

    fig = px.imshow(
        combined,
        animation_frame=0,
        facet_col=1,
        binary_string=False,
        color_continuous_scale="Viridis",
        range_color=vrange,
        labels={"facet_col": "Type", "animation_frame": "Step"},
        facet_col_wrap=3,
    )

    # Update facet labels
    labels = ["Ground Truth", "Prediction", "Abs Error / std"]
    for i, label in enumerate(labels):
        if i < len(fig.layout.annotations):
            fig.layout.annotations[i].text = label

    # Update title dynamically for each frame to show precomputed RMSE
    for i, frame in enumerate(fig.frames):
        current_rmse = rmse_per_step[i]
        frame.layout.title = {
            "text": f"Step {i} | RMSE: {current_rmse:.5f} | Channel {channel_idx}",
            "x": 0.5,
            "xanchor": "center",
        }

    # Initial layout setup
    fig.update_layout(
        title={"text": f"Step 0 | RMSE: {rmse_per_step[0]:.5f} | Channel {channel_idx}", "x": 0.5, "xanchor": "center"},
        height=550,
    )

    fig.show()


def animate_results_mpl(pred_list, gt_list, channel_idx=0, vmin=None, vmax=None, fps=10):

    max_steps = min(pred_list.shape[0], gt_list.shape[0])
    p_all = pred_list[:max_steps, channel_idx].detach().cpu().numpy()
    g_all = gt_list[:max_steps, channel_idx].detach().cpu().numpy()

    std_g = np.std(g_all) + 1e-8
    error_all = np.abs(g_all - p_all) / std_g

    # RMSE
    rmse_per_step = np.sqrt(np.mean((g_all - p_all) ** 2, axis=(1, 2)))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.subplots_adjust(wspace=0.4)

    v_min = vmin if vmin is not None else np.min(g_all)
    v_max = vmax if vmax is not None else np.max(g_all)

    im1 = ax1.imshow(g_all[0], cmap="viridis", vmin=v_min, vmax=v_max)
    ax1.set_title("Ground Truth")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    im2 = ax2.imshow(p_all[0], cmap="viridis", vmin=v_min, vmax=v_max)
    ax2.set_title("Prediction")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    im3 = ax3.imshow(error_all[0], cmap="Reds", vmin=0, vmax=np.max(error_all))
    ax3.set_title("Abs Error / std")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    def update(frame):
        im1.set_data(g_all[frame])
        im2.set_data(p_all[frame])
        im3.set_data(error_all[frame])
        fig.suptitle(f"Step {frame} | RMSE: {rmse_per_step[frame]:.5f}", fontsize=16)
        return im1, im2, im3

    ani = FuncAnimation(fig, update, frames=len(g_all), interval=1000 / fps, blit=True)

    plt.show()
    # Pour sauvegarder :
    # ani.save('resultats.mp4', writer='ffmpeg')
    return ani


def plot_comparison(pred_list, gt_list, RMSE_list, n_images=5, start_idx=2, channel_idx=0):
    """
    Display N predicted vs Ground Truth images for a specific channel.

    Args:
        pred_list: List of predicted tensors [C, H, W]
        gt_list: List of ground truth tensors [C, H, W]
        n_images: Number of frames to show
        start_idx: The time index of the first prediction (for labeling)
        channel_idx: Which channel to visualize (default: 0)
    """
    n = min(n_images, len(pred_list), len(gt_list))

    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6))

    # Handle the case where n=1 (axes is not a 2D array)
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        # Extract specific channel and move to CPU
        # Shape: [H, W]
        p_img = pred_list[start_idx + i][channel_idx].detach().cpu()
        if start_idx + i < gt_list.shape[0]:
            g_img = gt_list[start_idx + i][channel_idx].detach().cpu()
        else:
            g_img = np.zeros_like(p_img)

        # Row 0: Predictions
        axes[0, i].imshow(p_img, cmap="viridis")
        axes[0, i].set_title(f"Pred (t={start_idx + i}, RMSE={RMSE_list[start_idx+i]:.3f})")
        axes[0, i].axis("off")

        # Row 1: Ground Truth
        axes[1, i].imshow(g_img, cmap="viridis")
        axes[1, i].set_title(f"True (t={start_idx + i})")
        axes[1, i].axis("off")

    plt.suptitle(f"Comparison for Channel {channel_idx}", fontsize=16)
    plt.tight_layout()
    plt.show()
