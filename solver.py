import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import pandas as pd
from tqdm import tqdm
from scipy.sparse.linalg import factorized
from visualize import plot_matrix



def idx(i, j, Nx, Ny):
    return i * Ny + j


def build_A_robin(Lx, Ly, Nx, Ny, Bi):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    
    invdx2 = 1.0 / (dx * dx)
    invdy2 = 1.0 / (dy * dy)

    total_nodes = Nx * Ny
    A = sp.lil_matrix((total_nodes, total_nodes), dtype=float)

    extra_center_x = -(2.0 * dx * Bi) * invdx2
    extra_center_y = -(2.0 * dy * Bi) * invdy2

    for i in range(Nx):
        for j in range(Ny):
            p = idx(i, j, Nx, Ny)
            center = 0.0

            # ---- x-retning (i-indeks) ----
            if i == 0:
                # Venstre rand
                A[p, idx(1, j, Nx, Ny)] += 2.0 * invdx2
                center += (-2.0 * invdx2) + extra_center_x
            elif i == Nx - 1:
                # Høyre rand
                A[p, idx(Nx - 2, j, Nx, Ny)] += 2.0 * invdx2
                center += (-2.0 * invdx2) + extra_center_x
            else:
                # Indre punkter i x
                A[p, idx(i - 1, j, Nx, Ny)] += invdx2
                A[p, idx(i + 1, j, Nx, Ny)] += invdx2
                center += -2.0 * invdx2

            # ---- y-retning (j-indeks) ----
            if j == 0:
                # Nedre rand
                A[p, idx(i, 1, Nx, Ny)] += 2.0 * invdy2
                center += (-2.0 * invdy2) + extra_center_y
            elif j == Ny - 1:
                # Øvre rand
                A[p, idx(i, Ny - 2, Nx, Ny)] += 2.0 * invdy2
                center += (-2.0 * invdy2) + extra_center_y
            else:
                # Indre punkter i y
                A[p, idx(i, j - 1, Nx, Ny)] += invdy2
                A[p, idx(i, j + 1, Nx, Ny)] += invdy2
                center += -2.0 * invdy2

            # Sett diagonalelementet
            A[p, p] = center

    return A.tocsr()



def solve_heateq_backward_euler(Lx, Ly, Nx, Ny, tm_end, cfg, dt=0.01):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    Nt_steps = int(np.ceil(tm_end/dt))

    A = build_A_robin(Lx, Ly, Nx, Ny, cfg.Bi)

    total_nodes = Nx * Ny

    I = sp.eye(total_nodes, format='csr')
    implicit_op = I - dt * A
    
    solve = factorized(implicit_op)

    u = -np.ones(total_nodes, dtype=float)

    overheated = False
    
    center_idx = (Nx // 2) * Ny + (Ny // 2)

    results = {
        "time": [], 
        "u": [], 
        "u_max": [], 
        "center_u": [], 
        "Overheat": [overheated]
    }

    for n in range(Nt_steps + 1):

        results["time"].append(n * dt)
        results["u"].append(u.reshape(Nx, Ny).copy())
        results["u_max"].append(u.max())
        results["center_u"].append(u[center_idx])
        

        if u.max() > 0 and not overheated:
            overheated = True
            results["Overheat"][0] = True

        u = solve(u)

    return results


def solve_heateq_alpha(Lx, Ly, Nx, Ny, tm_end, cfg, dt=0.001):
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)

    Nt_steps = int(np.ceil(tm_end/dt))

    A = build_A_robin(Lx, Ly, Nx, Ny, cfg.Bi)

    total_nodes = Nx * Ny
    I = sp.eye(total_nodes, format='csr')
    implicit_op = I - dt * A

    u = -np.ones(total_nodes, dtype=float)
    alpha = np.full(total_nodes, cfg.alpha_initial, dtype=float)
    
    dalpha_dt = cfg.Da * np.exp(-cfg.gamma / (1 + cfg.epsilon * u))

    overheated = False
    
    center_idx = (Nx // 2) * Ny + (Ny // 2)

    results = {
        "time": [], 
        "u": [], 
        "u_max": [], 
        "alpha": [], 
        "alpha_max": [], 
        "dalpha_dt": [], 
        "dalpha_dt_max": [], 
        "center_alpha": [], 
        "center_u": [], 
        "Overheat": [overheated]
    }
    
    for n in range(Nt_steps + 1):
        Q = cfg.Ex * dalpha_dt
        rhs = u + dt * Q

        u_new = spsolve(implicit_op, rhs)

        dalpha_dt = cfg.update_alpha(u, alpha)
        alpha_new = alpha + dt * dalpha_dt
        alpha_new = np.clip(alpha_new, 0, 1.0) 

        u, alpha = u_new, alpha_new

        results["time"].append(n * dt)
        results["u"].append(u.reshape(Nx, Ny).copy())
        results["u_max"].append(u.max())
        results["alpha"].append(alpha.reshape(Nx, Ny).copy())
        results["alpha_max"].append(alpha.max())
        results["dalpha_dt"].append(dalpha_dt.reshape(Nx, Ny).copy())
        results["dalpha_dt_max"].append(dalpha_dt.max())
        results["center_alpha"].append(alpha[center_idx])
        results["center_u"].append(u[center_idx])
        
        if np.max(u) > 0 and not overheated:
            overheated = True
            results["Overheat"][0] = True

    return results



def idx_3d(i, j, k, Nx, Ny, Nz):
    return i * (Ny * Nz) + j * Nz + k


def build_A_3d(Lx, Ly, Lz, Nx, Ny, Nz, Bi_side, Bi_top, Bi_bottom):
    dx, dy, dz = Lx/(Nx-1), Ly/(Ny-1), Lz/(Nz-1)
    invdx2, invdy2, invdz2 = 1/dx**2, 1/dy**2, 1/dz**2
    
    total_nodes = Nx * Ny * Nz
    A = sp.lil_matrix((total_nodes, total_nodes), dtype=float)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                p = idx_3d(i, j, k, Nx, Ny, Nz)
                center = 0.0

                # --- X-retning (Sider) ---
                if i == 0 or i == Nx-1:
                    neighbor = idx_3d(1 if i==0 else Nx-2, j, k, Nx, Ny, Nz)
                    A[p, neighbor] += 2.0 * invdx2
                    center += -2.0 * invdx2 - (2.0 * dx * Bi_side) * invdx2
                else:
                    A[p, idx_3d(i-1, j, k, Nx, Ny, Nz)] += invdx2
                    A[p, idx_3d(i+1, j, k, Nx, Ny, Nz)] += invdx2
                    center -= 2.0 * invdx2

                # --- Y-retning (Sider) ---
                if j == 0 or j == Ny-1:
                    neighbor = idx_3d(i, 1 if j==0 else Ny-2, k, Nx, Ny, Nz)
                    A[p, neighbor] += 2.0 * invdy2
                    center += -2.0 * invdy2 - (2.0 * dy * Bi_side) * invdy2
                else:
                    A[p, idx_3d(i, j-1, k, Nx, Ny, Nz)] += invdy2
                    A[p, idx_3d(i, j+1, k, Nx, Ny, Nz)] += invdy2
                    center -= 2.0 * invdy2

                # --- Z-retning (Topp / Bunn) ---
                if k == 0: # Bunn (Stein/Metall)
                    A[p, idx_3d(i, j, 1, Nx, Ny, Nz)] += 2.0 * invdz2
                    center += -2.0 * invdz2 - (2.0 * dz * Bi_bottom) * invdz2
                elif k == Nz-1: # Topp (Luft)
                    A[p, idx_3d(i, j, Nz-2, Nx, Ny, Nz)] += 2.0 * invdz2
                    center += -2.0 * invdz2 - (2.0 * dz * Bi_top) * invdz2
                else:
                    A[p, idx_3d(i, j, k-1, Nx, Ny, Nz)] += invdz2
                    A[p, idx_3d(i, j, k+1, Nx, Ny, Nz)] += invdz2
                    center -= 2.0 * invdz2

                A[p, p] = center

    return A.tocsr()



def solve_heateq_alpha_3d(Lx, Ly, Lz, Nx, Ny, Nz, tm_end, cfg, Bi_side, Bi_top, Bi_bottom, dt=0.001):
    
    A = build_A_3d(Lx, Ly, Lz, Nx, Ny, Nz, Bi_side, Bi_top, Bi_bottom)

    total_nodes = Nx * Ny * Nz
    I = sp.eye(total_nodes, format='csr')
    implicit_op = I - dt * A

    u = -np.ones(total_nodes, dtype=float)
    alpha = np.full(total_nodes, cfg.alpha_initial, dtype=float)
    
    center_idx = idx_3d(Nx // 2, Ny // 2, Nz // 2, Nx, Ny, Nz)

    results = {
        "time": [], 
        "u_max": [], 
        "alpha_max": [], 
        "center_u": [],
        "center_alpha": [],
        "xy_slice": [], 
        "xz_slice": [], 
        "Overheat": [False]
    }

    nt_steps = int(np.ceil(tm_end/dt))
    
    for n in range(nt_steps + 1):

        dalpha_dt = cfg.update_alpha(u, alpha)
        
        Q = cfg.Ex * dalpha_dt
        rhs = u + dt * Q

        u_new = spsolve(implicit_op, rhs)
        
        alpha = np.clip(alpha + dt * dalpha_dt, 0, 1.0)
        u = u_new

        if n % 5 == 0:
            u_3d = u.reshape(Nx, Ny, Nz)
            results["time"].append(n * dt)
            results["u_max"].append(u.max())
            results["alpha_max"].append(alpha.max())
            results["center_u"].append(u[center_idx])
            results["center_alpha"].append(alpha[center_idx])
            results["xy_slice"].append(u_3d[:, :, Nz // 2].copy())
            results["xz_slice"].append(u_3d[:, Ny // 2, :].copy())

        if u.max() > 0 and not results["Overheat"][0]:
            results["Overheat"][0] = True

    return results