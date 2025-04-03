import numpy as np

def select_waypoint(x, waypoints, curr_idx, threshold=0.3):
    if curr_idx >= len(waypoints) or curr_idx==-1:
        return waypoints[-1], curr_idx  # Stay at last waypoint
    distance = np.linalg.norm(waypoints[curr_idx] - x[:2])
    if distance < threshold and curr_idx < len(waypoints) - 1:
        curr_idx += 1
        target_wp = np.array(waypoints[curr_idx])
    elif distance < threshold and curr_idx==len(waypoints) - 1:
        target_wp = np.array(waypoints[curr_idx])
        curr_idx = -1
    else:
        target_wp = np.array(waypoints[curr_idx])
        curr_idx = curr_idx
    return target_wp, curr_idx

def solve_sequence(x0, dynamics, nt, dt, u_min, u_max, 
                   waypoints, Kp, Ki, Kd, error_fn):
    trajs = [np.array(x0)]
    curr_idx = 0
    prev_p_error = None
    i_error = 0
    infos = {}
    us = []
    for ti in range(nt):
        x = trajs[-1]
        target_point, curr_idx = select_waypoint(x, waypoints, curr_idx)
        p_error = error_fn(x, target_point, curr_idx, waypoints)
        if prev_p_error is None:
            i_error = p_error
            d_error = np.zeros_like(p_error)
        else:
            i_error = i_error + p_error * dt
            d_error = (p_error - prev_p_error)/dt
        
        raw_u = Kp.T @ p_error + Ki.T @ i_error + Kd.T @ d_error
        u = np.minimum(np.maximum(raw_u, u_min), u_max)
        new_x = dynamics(x, u, dt)
        us.append(u)
        trajs.append(new_x)
        prev_p_error = p_error
    
    trajs = np.stack(trajs, axis=0)
    us = np.stack(us, axis=0)
    return trajs, us, infos