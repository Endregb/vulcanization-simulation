import numpy as np
from config import Config
from solver import solve_heateq_alpha, solve_heateq_backward_euler, build_A_robin, solve_heateq_alpha_3d
from analyze import analyze_cure, convergence_test_spatial, convergence_test_time, convergence_test_spatial2
from visualize import plot_animation_2d, plot_linear, plot_convergence_spatial, plot_convergence_time, plot_matrix, plot_snapshots_heateq



cfg = Config()

# --- 1. Plotting the matrix A ---
"""
Nx, Ny = 7, 7
Lx, Ly = 1, 1
Tend = 0.2
dt = 0.001
x, y = np.linspace(0, 100 * cfg.L, Nx), np.linspace(0, 100 * cfg.L, Ny)

A = build_A_robin(Lx, Ly, Nx, Ny, cfg.Bi)
plot_matrix(A, Nx)
"""


# --- 2. Vizulizing solution of heat equation with Tm = 443, T0 = 300 ---

"""
Nx, Ny = 200, 200
Lx, Ly = 1, 1
x, y = np.linspace(0, 100 * cfg.L, Nx), np.linspace(0, 100 * cfg.L, Ny)
Tend = 0.09
dt = 0.001


results = solve_heateq_backward_euler(Lx, Ly, Nx, Ny, Tend, cfg, dt)
t = np.array(results["time"]) * cfg.t_diff/360
T = cfg.Tm + np.array(results["u"]) * (cfg.Tm - cfg.T0)
T_max = cfg.Tm + np.array(results["u_max"]) * (cfg.Tm - cfg.T0)

plot_animation_2d(cfg=cfg, x=x, y=y, t=t, T=T)
plot_snapshots_heateq(cfg=cfg, x=x, y=y, t=t, T=T)
"""


# --- 3. Numerical convergence ---
"""
eoc_spatial, error_list_spatial, h_list_spatial = convergence_test_spatial(cfg, method="max")
plot_convergence_spatial(h_list_spatial, error_list_spatial, eoc_spatial, method="max")

eoc_spatial, error_list_spatial, h_list_spatial = convergence_test_spatial(cfg, method="mean")
plot_convergence_spatial(h_list_spatial, error_list_spatial, eoc_spatial, method="mean")

eoc_time, error_list_time, h_list_time = convergence_test_time(cfg)
plot_convergence_time(h_list_time, error_list_time, eoc_time)
"""


# --- 4. Including reaction term ---
"""
Nx, Ny = 100, 100
Lx, Ly = 1, 1
x, y = np.linspace(0, 100 * cfg.L, Nx), np.linspace(0, 100 * cfg.L, Ny) #Distance in centimeters
Tend = 0.2
dt = 0.001

results = solve_heateq_alpha(Lx, Ly, Nx, Ny, tm_end=Tend, cfg=cfg, dt=dt)

t = np.array(results["time"]) * cfg.t_diff/360
T = cfg.Tm + np.array(results["u"]) * (cfg.Tm - cfg.T0)
alpha = np.array(results["alpha"])
dt_alpha_max = np.array(results["dalpha_dt"])
T_max = cfg.Tm + np.array(results["u_max"]) * (cfg.Tm - cfg.T0)

plot_animation_2d(cfg=cfg, x=x, y=y, t=t, T=T)
plot_animation_2d(cfg=cfg, x=x, y=y, t=t, T=alpha)

plot_snapshots_heateq(cfg=cfg, x=x, y=y, t=t, T=T, title="Temperature", save_path="output/snapshots_heateq_with_alpha.png")
plot_snapshots_heateq(cfg=cfg, x=x, y=y, t=t, T=alpha, title="Alpha", save_path="output/snapshots_alpha")
"""


# --- 5. Big time interval analysis when including reaction term ---
"""
Nx, Ny = 50, 50
Lx, Ly = 1, 1
x, y = np.linspace(0, 100 * cfg.L, Nx), np.linspace(0, 100 * cfg.L, Ny) #Distance in centimeters
Tend = 40
dt = 0.05

results = solve_heateq_alpha(Lx, Ly, Nx, Ny, tm_end=Tend, cfg=cfg, dt=dt)

t = np.array(results["time"]) * cfg.t_diff/360
T_max = cfg.Tm + np.array(results["u_max"]) * (cfg.Tm - cfg.T0)
alpha_max = np.array(results["alpha_max"])
dt_alpha_max = np.array(results["dalpha_dt_max"])


plot_linear(t, T_max, title="Temperature", save_path="output/rate_of_temp.png")
plot_linear(t, alpha_max, title="Alpha", save_path="output/rate_of_alpha.png")
plot_linear(t, dt_alpha_max, title="dalpha", save_path="output/rate_of_dalpha.png")


T = cfg.Tm + np.array(results["u"]) * (cfg.Tm - cfg.T0)

plot_animation_2d(cfg=cfg, x=x, y=y, t=t, T=T)
"""


# --- 6. 90% cure ---
"""
Nx, Ny = 50, 50
Lx, Ly = 1, 1
x, y = np.linspace(0, 100 * cfg.L, Nx), np.linspace(0, 100 * cfg.L, Ny) #Distance in centimeters
Tend = 40
dt = 0.05

results = solve_heateq_alpha(Lx, Ly, Nx, Ny, tm_end=Tend, cfg=cfg, dt=dt)

analyze_cure(results, cfg)
"""


# --- 7. Rectangualar grid ---
"""
Lx = 2
Ly = 1
Nx = 160
Ny = 80
x, y = np.linspace(0, 100 * Lx, Nx), np.linspace(0, 100 * Ly, Ny) #Distance in centimeters


Tend = 0.2
dt = 0.001

results = solve_heateq_alpha(Lx=Lx, Ly=Ly, Nx=Nx, Ny=Ny, tm_end=Tend, cfg=cfg, dt=dt)
t = np.array(results["time"]) * cfg.t_diff/360
T = cfg.Tm + np.array(results["u"]) * (cfg.Tm - cfg.T0)

plot_animation_2d(cfg, x, y, t, T, save_path="S_code/output/animation_rect.png")
"""


# --- 8. 3d modeling, with robin conditions on top and bottom ---
"""
Lx = 2
Ly = 2
Lz = 1
Nx = 40
Ny = 40
Nz = 20


Tend = 0.1
dt = 0.001

Bi_side = 20
Bi_top = 2
Bi_bottom = 300

results = solve_heateq_alpha_3d(Lx, Ly, Lz, Nx, Ny, Nz, Tend, cfg, Bi_side, Bi_top, Bi_bottom,dt)

t = np.array(results["time"]) * cfg.t_diff/360
x = np.linspace(0, cfg.L*200, Nx)
y = np.linspace(0, cfg.L*200, Ny)
z = np.linspace(0, cfg.L*100, Nz)

xz_data = cfg.Tm + np.array(results["xz_slice"]) * (cfg.Tm - cfg.T0)
xy_data = cfg.Tm + np.array(results["xy_slice"]) * (cfg.Tm - cfg.T0)


plot_animation_2d(cfg, x, z, t, xz_data, title="Temperature XZ-slice",save_path="output/xz_animation.gif")
plot_snapshots_heateq(cfg, x, z, t, xz_data, title="Temperature XZ-slice", save_path="output/3d_xz_snapshots.png")
plot_snapshots_heateq(cfg, x, y, t, xy_data, title="Temperature XY-slice", save_path="output/3d_xy_snapshots.png")
"""


# --- 9. 3d modeling, curing analysis ---
"""
Lx = 2
Ly = 2
Lz = 1
Nx = 20
Ny = 20
Nz = 10


Tend = 40
dt = 0.05

Bi_side = 20
Bi_top = 2
Bi_bottom = 300

results = solve_heateq_alpha_3d(Lx, Ly, Lz, Nx, Ny, Nz, Tend, cfg, Bi_side, Bi_top, Bi_bottom,dt)

t = np.array(results["time"]) * cfg.t_diff/360

T_max = cfg.Tm + np.array(results["u_max"]) * (cfg.Tm - cfg.T0)
alpha_max = np.array(results["alpha_max"])

plot_linear(t, T_max, title="Temperature", save_path="output/rate_of_temp3d.png")
plot_linear(t, alpha_max, title="Alpha", save_path="output/rate_of_alpha3d.png")

analyze_cure(results, cfg)
"""

# --- 10. 3d modeling, find best geometry for fast curing ---

"""
Nx = 15
Ny = 15
Nz = 15

Lx_list = [1, 2, 3]
Ly_list = [1, 2, 3]
Lz_list = [1, 2, 3]

Tend = 40
dt = 0.1

Bi_side = 20
Bi_top = 2
Bi_bottom = 300

best_lengths = [0, 0, 0]
smalles_t = np.inf

for Lx in Lx_list:
    for Ly in Ly_list:
        for Lz in Lz_list:
            results = solve_heateq_alpha_3d(Lx, Ly, Lz, Nx, Ny, Nz, Tend, cfg, Bi_side, Bi_top, Bi_bottom,dt)
            t_90 = analyze_cure(results, cfg)
            if t_90 < smalles_t:
                smalles_t = t_90
                best_lengths = [Lx, Ly, Lz]

            print(Lx, Ly, Lz)

print(best_lengths)
"""