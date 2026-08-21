import numpy as np
from numba import njit
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


# ── JIT-compiled kernels ──────────────────────────────────────────────

@njit(cache=True)
def _alpha_ODE_jit(u, u_threshold, alpha, D, gamma, epsilon, m, n):
    """JIT-compiled alpha ODE kernel."""
    N = u.shape[0]
    result = np.empty(N)
    for i in range(N):
        if u[i] > u_threshold:
            result[i] = D * np.exp(-gamma / (1.0 + epsilon * u[i])) * alpha[i]**m * (1.0 - alpha[i])**n
        else:
            result[i] = 0.0
    return result


@njit(cache=True)
def _RK4_alpha_jit(u, alpha, dt_sub, u_threshold, n_sub, D, gamma, epsilon, m, n):
    """JIT-compiled RK4 sub-stepping."""
    dt_fine = dt_sub / n_sub
    alpha = alpha.copy()
    for _ in range(n_sub):
        k1 = _alpha_ODE_jit(u, u_threshold, alpha, D, gamma, epsilon, m, n)
        k2 = _alpha_ODE_jit(u, u_threshold, alpha + 0.5*dt_fine*k1, D, gamma, epsilon, m, n)
        k3 = _alpha_ODE_jit(u, u_threshold, alpha + 0.5*dt_fine*k2, D, gamma, epsilon, m, n)
        k4 = _alpha_ODE_jit(u, u_threshold, alpha + dt_fine*k3, D, gamma, epsilon, m, n)
        alpha = alpha + (dt_fine/6.0)*(k1 + 2.0*k2 + 2.0*k3 + k4)
        for i in range(alpha.shape[0]):
            if alpha[i] < 0.0:
                alpha[i] = 0.0
            elif alpha[i] > 1.0:
                alpha[i] = 1.0
    return alpha


# ── Public wrappers (same signature as solverE.py) ───────────────────

def alpha_ODE(cfg, u, u_threshold, alpha):
    return _alpha_ODE_jit(u, u_threshold, alpha, cfg.D, cfg.gamma, cfg.epsilon, cfg.m, cfg.n)


def RK4_alpha(cfg, u, alpha, dt_sub, u_threshold, n_sub=10):
    """Sub-step the alpha ODE with RK4 to handle stiffness."""
    return _RK4_alpha_jit(u, alpha, dt_sub, u_threshold, n_sub,
                          cfg.D, cfg.gamma, cfg.epsilon, cfg.m, cfg.n)


def solve_system(cfg, C_matrix, snap_interval):
    
    solve = factorized(C_matrix.tocsc())

    # Initial conditions
    alpha = np.ones(cfg.N**2) * cfg.params["alpha"]
    U = np.ones(cfg.N**2) * -1.0

    # History for animations and debugging
    history_U, history_alpha, history_alpha_rate = [], [], []

    steps = int(cfg.t_final/cfg.dt)
    for t in tqdm(range(steps), desc="Solving system (JIT)"):

        U_interim = solve(U)

        alpha_rate = alpha_ODE(cfg, U_interim, cfg.U_threshold, alpha)
        
        # Update alpha with RK4
        alpha = RK4_alpha(cfg, U_interim, alpha, cfg.dt, cfg.U_threshold)

        U = U_interim + cfg.dt * cfg.S * alpha_rate

        # Save for GIF
        if t % snap_interval == 0 or t == 0:
            history_U.append(U.copy())
            history_alpha.append(alpha.copy())
            history_alpha_rate.append(alpha_rate.copy())

    return history_U, history_alpha, history_alpha_rate
