import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm.notebook import tqdm
from mpl_toolkits.mplot3d import Axes3D


def GIF_maker(histories, titles, cmaps, vmins, vmaxs, filename, N, snap_interval, dt, fps=10):
    """
    Create an animated GIF from simulation histories.

    Parameters
    ----------
    histories : list of list of np.ndarray
        Each entry is a list of 1D snapshots (length N*N) over time.
    titles : list of str
        Subplot titles for each history.
    cmaps : list of str
        Colormaps for each subplot.
    vmins : list of float
        Min color values for each subplot.
    vmaxs : list of float
        Max color values for each subplot.
    filename : str
        Output GIF filename.
    N : int
        Grid size (N x N).
    snap_interval : int
        Number of timesteps between snapshots.
    dt : float
        Timestep size.
    fps : int
        Frames per second in the GIF.
    """
    n_plots = len(histories)
    n_frames = len(histories[0])

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Create initial images and colorbars
    imgs = []
    for i, ax in enumerate(axes):
        data = histories[i][0].reshape((N, N))
        im = ax.imshow(data, origin='lower', cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                        extent=[0, 1, 0, 1], aspect='equal')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        imgs.append(im)

    # Add a centered suptitle (will be updated each frame)
    suptitle = fig.suptitle('', fontsize=14, fontweight='bold')

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    def update(frame):
        t_dimensionless = frame * snap_interval * dt
        t_physical = frame * snap_interval * dt * 4180      # t_diff = 4180
        suptitle.set_text(f'Frame {frame + 1}/{n_frames}  |  t* = {t_dimensionless:.2f}s*  |  t = {t_physical:.2f}s')
        for i in range(n_plots):
            data = histories[i][frame].reshape((N, N))
            imgs[i].set_data(data)
            axes[i].set_title(f'{titles[i]}\nmin={data.min():.4f}, max={data.max():.4f}')
        return imgs

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"GIF saved as '{filename}' ({n_frames} frames, {fps} fps)")


def plot_3d(histories, titles, cmaps, vmins, vmaxs, N, snap_interval, dt, frame_idx=-1):
    """
    Plot 3D surface plots of the simulation fields at a given frame.

    Parameters
    ----------
    histories : list of list of np.ndarray
        Each entry is a list of 1D snapshots (length N*N) over time.
    titles : list of str
        Subplot titles for each history.
    cmaps : list of str
        Colormaps for each subplot.
    vmins : list of float
        Min color values for each subplot.
    vmaxs : list of float
        Max color values for each subplot.
    N : int
        Grid size (N x N).
    snap_interval : int
        Number of timesteps between snapshots.
    dt : float
        Timestep size.
    frame_idx : int
        Which frame to plot. Default -1 (last frame).
    """
    n_plots = len(histories)
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    t_physical = frame_idx * snap_interval * dt if frame_idx >= 0 else len(histories[0]) * snap_interval * dt

    fig = plt.figure(figsize=(8 * n_plots, 6))
    fig.suptitle(f'Frame {frame_idx}/{len(histories[0])}  |  t = {t_physical:.4f}',
                 fontsize=14, fontweight='bold')

    for i in range(n_plots):
        ax = fig.add_subplot(1, n_plots, i + 1, projection='3d')
        Z = histories[i][frame_idx].reshape((N, N))

        surf = ax.plot_surface(X, Y, Z, cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                               edgecolor='none', alpha=0.9)
        fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel(titles[i])
        ax.set_title(f'{titles[i]}\nmin={Z.min():.4f}, max={Z.max():.4f}')
        ax.set_zlim(vmins[i], vmaxs[i])
        ax.view_init(elev=30, azim=225)

    plt.tight_layout()
    plt.show()


def plot_3d_gif(histories, titles, cmaps, vmins, vmaxs, filename, N, snap_interval, dt, fps=10, elev=30, azim=225):
    """
    Create an animated 3D surface GIF.

    Parameters
    ----------
    histories : list of list of np.ndarray
    titles : list of str
    cmaps : list of str
    vmins, vmaxs : list of float
    filename : str
    N : int
    snap_interval : int
    dt : float
    fps : int
    elev, azim : float
        Camera angle for the 3D view.
    """
    n_plots = len(histories)
    n_frames = len(histories[0])

    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8 * n_plots, 6))
    suptitle = fig.suptitle('', fontsize=14, fontweight='bold')

    axes = []
    for i in range(n_plots):
        ax = fig.add_subplot(1, n_plots, i + 1, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlim(vmins[i], vmaxs[i])
        ax.view_init(elev=elev, azim=azim)
        axes.append(ax)

    def update(frame):
        t_physical = frame * snap_interval * dt
        suptitle.set_text(f'Frame {frame + 1}/{n_frames}  |  t = {t_physical:.4f}')
        for i in range(n_plots):
            axes[i].cla()
            Z = histories[i][frame].reshape((N, N))
            axes[i].plot_surface(X, Y, Z, cmap=cmaps[i], vmin=vmins[i], vmax=vmaxs[i],
                                 edgecolor='none', alpha=0.9)
            axes[i].set_xlabel('x')
            axes[i].set_ylabel('y')
            axes[i].set_zlabel(titles[i])
            axes[i].set_title(f'{titles[i]}\nmin={Z.min():.4f}, max={Z.max():.4f}')
            axes[i].set_zlim(vmins[i], vmaxs[i])
            axes[i].view_init(elev=elev, azim=azim)

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    anim.save(filename, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"3D GIF saved as '{filename}' ({n_frames} frames, {fps} fps)")