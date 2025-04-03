import torch

def solve_sequence(x0, dynamics, nt, dt, u_min, u_max, n_inits, n_iters, lr, loss_fn, quiet=False):
    info = {}
    udim = u_min.shape[0]
    us = (0.1 * torch.randn(n_inits, nt, udim)).requires_grad_(True)
    x = torch.from_numpy(x0)[None, :].repeat(n_inits, 1)
    optimizer = torch.optim.Adam([us], lr=lr)
    for iter_i in range(n_iters):
        x_list = [x]
        for ti in range(us.shape[-2]):
            new_x = dynamics(x_list[-1], us[..., ti, :], dt)
            x_list.append(new_x)
        trajs = torch.stack(x_list, dim=-2)
        losses = loss_fn(trajs, us, u_min, u_max)
        loss = torch.mean(losses)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if not quiet:
            print("%04d/%04d loss:%.3f"%(iter_i, n_iters, loss.item()))
    min_idx = torch.argmin(losses).item()
    trajs = (trajs[min_idx]).detach().cpu()
    us = (us[min_idx]).detach().cpu()
    
    return trajs, us, info