import os
import numpy as np


def compute_area_conservation_error(cfg, trial: int = 1, t_end: float = 8.0,
                                    r: float = 0.15):
    """
    Compute the area conservation error at the final time of each time window.

    The reference area is that of the initial circle: A_exact = pi * r^2.
    For each window the numerical area is integrated from the saved prediction
    file, and the relative error is reported as:

        error = |A_numerical - A_exact|
    """
    area_exact = np.pi * r ** 2

    filename = os.path.join(cfg.out_dir, f"{trial}_final_pred_{t_end:.2f}.dat")

    data_final = np.loadtxt(filename)
    area_numerical = np.sum(data_final[:, -1]) * 1.0 / ( (cfg.dom_dis[0]-1)*(cfg.dom_dis[1]-1) )
    error = abs(area_numerical - area_exact)
    return error


'''
def compute_area_conservation_error(cfg, trial: int = 1,
                                    r: float = 0.15):
    """
    Compute the area conservation error at the final time of each time window.

    The reference area is that of the initial circle: A_exact = pi * r^2.
    For each window the numerical area is integrated from the saved prediction
    file, and the relative error is reported as:

        error = |A_numerical - A_exact|
    """
    area_exact = np.pi * r ** 2

    errors = []
    for i in range(cfg.num_win):
        t_end = cfg.state_t_array[i + 1]
        filename = os.path.join(cfg.out_dir, f"{trial}_final_pred_{t_end:.2f}.dat")
        print(f'[area] loading t = {t_end:.2f}  ->  {filename}')

        data_final = np.loadtxt(filename)
        area_numerical = np.sum(data_final[:, -1]) * 1.0 / ( (cfg.dom_dis[0]-1)*(cfg.dom_dis[1]-1) )
        error = abs(area_numerical - area_exact)
        print(f'[area] t = {t_end:.2f}  |  A_num = {area_numerical:.6e}  |  '
              f'A_exact = {area_exact:.6e}  |  error = {error:.6e}')
        errors.append(error)

    errors = np.array(errors)
    out_path = os.path.join(cfg.out_dir, f"{trial}_area_conservation_error.dat")
    np.savetxt(out_path, errors)
    print(f'[area] saved to {out_path}')

    return errors
'''
