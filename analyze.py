import numpy as np
from tqdm import tqdm

from solver import solve_heateq_backward_euler

def analyze_cure(results, cfg):
    times = np.array(results["time"])
    center_alphas = np.array(results["center_alpha"])
    
    # Find index where alpha >= 0.9
    idx_90 = np.where(center_alphas >= 0.9)[0]
    
    if len(idx_90) > 0:
        t_90 = times[idx_90[0]] * cfg.t_diff / 360
        print(f"Time to reach 90% cure in center: {t_90:.4f} hours")
    else:
        print("90% cure not reached within the simulation time.")
    
    return t_90




def convergence_test_spatial(cfg, method="max"):
    t_test = 0.2
    dt = 0.0001
    Lx = Ly = 1.0
    
    grid_list = np.array([5, 10, 20, 40, 80])
    
    N_ref = 300
    print(f"Genererer referanseløsning med N={N_ref}...")
    fasit_result = solve_heateq_backward_euler(Lx, Ly, N_ref, N_ref, tm_end=t_test, cfg=cfg, dt=dt)
    if method == "max":
        u_fasit_final = fasit_result["u"][-1].max()
    if method == "mean":
        u_fasit_final = fasit_result["u"][-1].mean()


    error_list = []

    for N in grid_list:
        print(f"Tester N = {N}...")
        result = solve_heateq_backward_euler(Lx, Ly, N, N, tm_end=t_test, cfg=cfg, dt=dt)
        
        if method == "max":
            u_final = result["u"][-1].max()
        if method == "mean":
            u_final = result["u"][-1].mean()
        
        error = np.abs(u_final - u_fasit_final)
        error_list.append(error)
    
    error_list = np.array(error_list)
    h_list = 1.0 / (grid_list - 1) 

    eoc = np.log(error_list[1:] / error_list[:-1]) / np.log(h_list[1:] / h_list[:-1])

    print(f"EOC: {eoc}")
    return eoc, error_list, h_list


    


def convergence_test_time(cfg):
    t_end = 0.2
    n_list = np.array([32, 64, 128, 256, 512])
    h_list = t_end/n_list

    Nx, Ny = 100, 100
    Lx, Ly = 1, 1

    fasit_results = solve_heateq_backward_euler(Lx, Ly, Nx, Ny, tm_end=t_end, cfg=cfg, dt=t_end/6000)
    center_u_fasit = np.max(fasit_results["u"])

    error_list = []
    for dt in h_list:
        result = solve_heateq_backward_euler(Lx, Ly, Nx, Ny, tm_end=t_end, cfg=cfg, dt=dt)
        center_u = np.max(result["u"])
        error = np.abs(center_u - center_u_fasit)
        error_list.append(error)
    

    error_list = np.array(error_list)
    h_list = np.array(h_list)

    eoc = np.log(error_list[1:]/error_list[:-1])/np.log(h_list[1:]/h_list[:-1])

    print(f"EOC: {eoc}")

    return eoc, error_list, h_list




