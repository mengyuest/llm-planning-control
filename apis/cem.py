import numpy as np

def solve_sequence(x0, dynamics, nt, dt, u_min, u_max, n_popsize, n_elites, n_iters, loss_fn, quiet=False):
    info = {}
    udim = u_min.shape[0]

    x = np.tile(x0[None, :], (n_popsize, 1))
    mean = np.zeros((nt, udim))
    std = np.ones((nt, udim))

    for iter_i in range(n_iters):
        us = mean + std * np.random.randn(n_popsize, nt, udim)
        us = np.clip(us, u_min, u_max)

        x_list = [x]
        for ti in range(us.shape[-2]):
            new_x = dynamics(x_list[-1], us[..., ti, :], dt)
            x_list.append(new_x)
        trajs = np.stack(x_list, axis=-2)

        scores = loss_fn(trajs, us, u_min, u_max)
        elite_idxs = np.argsort(scores)[:n_elites]
        elite_samples = us[elite_idxs]
        mean = elite_samples.mean(axis=0)
        std = 0.9 * std + 0.1 * elite_samples.std(axis=0) + 1e-6
        if not quiet:
            print("%04d/%04d loss:%.3f"%(iter_i, n_iters, scores[elite_idxs[0]]))
    
    min_idx = np.argmin(scores)
    return trajs[min_idx], us[min_idx], info