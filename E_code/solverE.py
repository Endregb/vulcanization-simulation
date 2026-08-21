import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
from tqdm.notebook import tqdm



def assemble_matrix(cfg):
    def tridiag(v, d, w, N):
        # Helper function
        # Returns tridiagonal matrix with v as lower diagonal, d as diagonal, w as upper diagonal 
        e = np.ones(N)
        A = v*np.diag(e[1:], -1) + d*np.diag(e) + w*np.diag(e[1:], 1)
        return A
    
    I = np.eye(cfg.N)
    I2 = np.eye(cfg.N**2)

    dn = 1.0 / (cfg.M)  # step size in space
    coeff = cfg.dt / dn**2

    # Create matrix for inner points
    B_matrix = tridiag(1, -4, 1, cfg.N)
    A_matrix = np.kron(I, B_matrix) + np.kron(B_matrix, I) + 4*I2
    C_matrix = np.eye(cfg.N**2) - (coeff) * A_matrix

    # Find indices of boundary nodes
    indices = np.arange(cfg.N**2).reshape((cfg.N, cfg.N))
    left_mask = indices[:, 0] # last row in each block
    right_mask = indices[:, -1] # first row in each block
    top_mask = indices[-1, :] # last block
    bottom_mask = indices[0, :] # first block

    # Alter current node and node opposite to ghost node
    for k in left_mask:
        C_matrix[k, k] += coeff * 2*dn*cfg.Bi        # current node
        C_matrix[k, k + 1] -= coeff                  # east node

    for k in right_mask:
        C_matrix[k, k] += coeff * 2*dn*cfg.Bi        # current node
        C_matrix[k, k - 1] -= coeff                  # west node

    for k in top_mask:
        C_matrix[k, k] += coeff * 2*dn*cfg.Bi        # current node
        C_matrix[k, k - cfg.N] -= coeff              # south node

    for k in bottom_mask:
        C_matrix[k, k] += coeff * 2*dn*cfg.Bi        # current node
        C_matrix[k, k + cfg.N] -= coeff              # north node

    return csr_matrix(C_matrix)

def alpha_ODE(cfg, u, u_threshold, alpha):
    # NB! Suspect something is wrong with this
    alpha_rate = cfg.D * np.exp(- cfg.gamma/(1 + cfg.epsilon * u)) * alpha**cfg.m * (1-alpha)**cfg.n
    return np.where(u > u_threshold, alpha_rate, 0)


def RK4_alpha(cfg, u, alpha, dt_sub, u_threshold, n_sub=10):
    """Sub-step the alpha ODE with RK4 to handle stiffness."""
    # dt_fine = dt_sub
    dt_fine = dt_sub / n_sub  # finer timestep to handle the stiffness of the Arrhenius
    for _ in range(n_sub):
        k1 = alpha_ODE(cfg, u, u_threshold, alpha)
        k2 = alpha_ODE(cfg, u, u_threshold, alpha + 0.5*dt_fine*k1)
        k3 = alpha_ODE(cfg, u, u_threshold, alpha + 0.5*dt_fine*k2)
        k4 = alpha_ODE(cfg, u, u_threshold, alpha + dt_fine*k3)
        alpha = alpha + (dt_fine/6)*(k1 + 2*k2 + 2*k3 + k4)
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha

def solve_system(cfg, C_matrix, snap_interval):
    
    solve = factorized(C_matrix.tocsc()) # convert to csc matrix to avoid SparseEfficienyWarning

    # Initial conditions
    alpha = np.ones(cfg.N**2) * cfg.params["alpha"]
    U = np.ones(cfg.N**2) * -1.0

    # History for animations and debugging
    history_U, history_alpha, history_alpha_rate = [], [], []

    steps = int(cfg.t_final/cfg.dt) # number of timesteps
    for t in tqdm(range(steps), desc="Solving system"):

        U_interim = solve(U)

        alpha_rate = alpha_ODE(cfg, U_interim, cfg.U_threshold, alpha)
        
        # # Update alpha with RK4
        alpha = RK4_alpha(cfg, U_interim, alpha, cfg.dt, cfg.U_threshold)

        # # Update alpha with Newton
        # alpha = alpha + cfg.dt*alpha_rate

        U = U_interim + cfg.dt * cfg.S * alpha_rate

        # Save for GIF and debug
        if t % snap_interval == 0 or t == 0:
            history_U.append(U.copy())
            history_alpha.append(alpha.copy())
            history_alpha_rate.append(alpha_rate.copy())

    return history_U, history_alpha, history_alpha_rate