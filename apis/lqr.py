import numpy as np
import scipy.linalg

def solve_sequence(x0, dynamics, linearize, Q, R, dt=0.1, T=100, ref_states=None, ref_inputs=None):
    x_dim = len(x0)
    u_dim = linearize(x0, np.zeros(1), dt)[1].shape[1]
    info = {}

    assert ref_states is not None
    if ref_inputs is None:
        ref_inputs = np.zeros((T, u_dim))
        for k in range(T - 1):
            A, B = linearize(ref_states[k], np.zeros(u_dim), dt)
            delta = ref_states[k + 1] - A @ ref_states[k]
            ref_inputs[k], _, _, _ = np.linalg.lstsq(B, delta, rcond=None)
        ref_inputs[-1] = ref_inputs[-2]

    trajs = np.zeros((T, x_dim))
    us = np.zeros((T, u_dim))
    trajs[0] = x0

    for k in range(T - 1):
        A, B = linearize(ref_states[k], ref_inputs[k], dt)
        X = scipy.linalg.solve_discrete_are(A, B, Q, R)
        K = np.linalg.inv(B.T @ X @ B + R) @ (B.T @ X @ A)
        u = ref_inputs[k] - K @ (trajs[k] - ref_states[k])
        trajs[k + 1] = dynamics(trajs[k], u, dt)
        us[k] = u

    us[-1] = us[-2]
    info["ref_states"] = ref_states 
    info["ref_inputs"] = ref_inputs
    return trajs, us, info