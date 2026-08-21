import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import os


def plot_snapshots_heateq(
    cfg, x, y, t, T, 
    title="Temperature", 
    cmap="hot", 
    save_path="output/snapshots_heateq.png"
):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    indices = np.linspace(0, len(t) - 1, 8, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 9), sharex=True, sharey=True)
    axes = axes.flatten() 
    
    X, Y = np.meshgrid(x, y, indexing="ij")
    vmin, vmax = T.min(), T.max()
    
    print(f"Genererer 8 snapshots til {save_path}...")

    for i, idx in enumerate(indices):
        ax = axes[i]
        
        im = ax.pcolormesh(X, Y, T[idx], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        
        mid_x, mid_y = T.shape[1] // 2, T.shape[2] // 2
        center_val = T[idx, mid_x, mid_y]
        
        ax.set_title(f"t = {t[idx]:.2f} \nCenter: {center_val:.1f}")

        if i >= 4: ax.set_xlabel("x (cm)")
        if i % 4 == 0: ax.set_ylabel("y (cm)")
        
        ax.set_aspect("equal")

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.1)
    cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label=title)
    
    plt.suptitle(f"Time evolution of {title}", fontsize=16, y=0.95)

    # 4. Lagre og vis
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Ferdig! Lagret i: {save_path}")



def plot_matrix(A, N, save_path = "output/matrix_A.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(A.todense(), cmap='coolwarm') 
    plt.colorbar(label='Value')
    plt.title(f"Values of A, with gridsize $N+1 = $ {N}")

    plt.savefig(save_path)
    plt.show()



def plot_animation_2d(
    cfg, x, y, t, T,
    title="Temperature",
    cmap="hot",
    save_path="output/animation.gif",
    fps=10,
):
    """Create animation and save 100 PNG snapshots for LaTeX"""
    
    snapshot_dir = "output/animation_heateq"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    X, Y = np.meshgrid(x, y, indexing="ij")

    vmin, vmax = T.min(), T.max()
    im = ax.pcolormesh(X, Y, T[0], shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_aspect("equal")

    plt.colorbar(im, label=title)
    
    mid_x = T.shape[1] // 2
    mid_y = T.shape[2] // 2
    spiked_at = ""

    def update(frame):
        nonlocal spiked_at
        im.set_array(T[frame].ravel())
        center_temp = T[frame, mid_x, mid_y]

        if center_temp > cfg.Tm and center_temp > cfg.max_temp:
            cfg.max_temp = center_temp
            spiked_at = f"\nSpiked at {t[frame]:.2f}, with value {cfg.max_temp:.2f}"

        ax.set_title(f"{title} at t = {t[frame]:.2f} h, center {center_temp:.2f}{spiked_at}")
        return [im]

    anim = FuncAnimation(fig, update, frames=len(t), interval=1000 / fps, blit=False)
    plt.show()




def plot_linear(x, y, title="Alpha", save_path="output/alpha/rate_of_alpha.png"):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker='o', linestyle='-') 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Time [s]")
    if title == "Temperature":
        plt.ylabel("Max temp T on grid")
        Tm = np.ones(len(x)) * 443
        plt.plot(x, Tm)
        plt.legend(("Temp", "Mold temp"))
    if title == "Alpha":
        plt.ylabel("Alpha in center")
    if title == "dalpha":
        plt.ylabel("dt_alpha in center")
    plt.title(f"Time variation of {title}")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to: {save_path}")
    plt.show()

    plt.close()




def plot_convergence_spatial(h_list, error_list, eoc, method="max"):
    plt.figure(figsize=(10, 7))
    
    # Choose label based on the error measurement method
    if method == "max":
        error_label = 'Numerical Error ($L_\infty$ / Max)'
        method_title = "Local (Max)"
    else:
        error_label = 'Numerical Error ($L_2$ / Mean)'
        method_title = "Global (Mean)"
    
    # 1. Plot the actual numerical error
    plt.loglog(h_list, error_list, 'o-', label=error_label, linewidth=2, markersize=8)
    if method == "max":
    # 2. Create reference line for Order 2 (O(h^2))
    # We anchor the line in the middle of the data for better visual alignment
        mid = len(h_list) // 2
        C2 = error_list[mid] / (h_list[mid]**2)
        order_2_line = C2 * (h_list**2)
        plt.loglog(h_list, order_2_line, '--', color='gray', alpha=0.7, label='Theoretical Order 2 ($O(h^2)$)')
    
    # 3. Create reference line for Order 1 (O(h))
    if method == "mean":
        mid = len(h_list) // 2
        C1 = error_list[mid] / (h_list[mid]**1)
        order_1_line = C1 * (h_list**1)
        plt.loglog(h_list, order_1_line, ':', color='red', alpha=0.7, label='Theoretical Order 1 ($O(h)$)')
    
    # Formatting the plot
    plt.xlabel('Grid size $h$ (log scale)')
    plt.ylabel('Absolute Error (log scale)')
    plt.title(f'Spatial Convergence Analysis ({method_title})\nFinal EOC = {eoc[-1]:.2f}')
    
    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend()
    
    # Add text with all EOC values at the bottom
    eoc_text = "Estimated Order of Convergence (EOC): " + ", ".join([f"{val:.2f}" for val in eoc])
    plt.figtext(0.5, 0.02, eoc_text, wrap=True, horizontalalignment='center', 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Save the plot with the method name in the filename
    filename = f'output/convergence_spatial_{method}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved as {filename}")
    
    plt.show()
    plt.close()


def plot_convergence_time(dt_list, error_list, eoc):
    plt.figure(figsize=(10, 6))
    
    # Ensure inputs   arrays for calculation
    dt_list = np.array(dt_list)
    error_list = np.array(error_list)
    
    # Plot the actual numerical error (red line with dots for time)
    plt.loglog(dt_list, error_list, 'o-', color='tab:red', label='Numerical Error ($L_\infty$)', linewidth=2)
    
    # Create a reference line for Order 1: Error = C * dt^1
    # Implicit Euler is first-order accurate in time
    C = error_list[0] / (dt_list[0]**1)
    order_1_line = C * (dt_list**1)
    
    plt.loglog(dt_list, order_1_line, '--', color='gray', label='Theoretical Order 1 ($O(k)$)')
    
    # Formatting the plot
    plt.xlabel('Time step $\Delta t$ (log scale)')
    plt.ylabel('Absolute error (log scale)')
    plt.title(f'Temporal Convergence Analysis: Final EOC = {eoc[-1]:.2f}')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    # Add text with all EOC values at the bottom
    eoc_text = "Estimated Order of Convergence (EOC): " + ", ".join([f"{val:.2f}" for val in eoc])
    plt.figtext(0.5, 0.01, eoc_text, wrap=True, horizontalalignment='center', fontsize=10)
    
    # Save the plot to the current folder
    plt.savefig('output/convergence_time.png', dpi=300, bbox_inches='tight')
    print("Temporal convergence plot saved as convergence_time.png")
    
    plt.show()
    plt.close()

