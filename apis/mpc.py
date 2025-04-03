import casadi as ca

def solve_sequence(x0: ca.DM,
                  dynamics: ca.Function,
                  nt: int,
                  dt: float,
                  u_min: ca.DM,
                  u_max: ca.DM,
                  is_linear: bool,
                  objective_fn: ca.Function,
                  eq_constraint_fn_list: list[ca.Function],
                  ineq_constraint_fn_list: list[ca.Function], 
                  ):
    info = {}
    if is_linear:
        opti = ca.Opti("conic")
    else:
        opti = ca.Opti()
    
    state_dim = x0.shape[0]
    u_dim = u_min.shape[0]
    
    X = opti.variable(state_dim, nt+1)  # State trajectory
    U = opti.variable(u_dim, nt)  # Control trajectory
    
    # initial value setup
    opti.subject_to(X[:, 0] == x0)
    
    # control constraints
    for ti in range(nt):
        opti.subject_to(U[:, ti] >= u_min)
        opti.subject_to(U[:, ti] <= u_max)
    
    # dynamical constraints
    for ti in range(nt):
        opti.subject_to(dynamics(X[:, ti], U[:, ti], dt)==X[:, ti+1])
    
    # external constraints
    for eq_constraint_fn in eq_constraint_fn_list:
        opti.subject_to(eq_constraint_fn(X, U) == 0)
    
    for ineq_constraint_fn in ineq_constraint_fn_list:
        opti.subject_to(ca.vec(ineq_constraint_fn(X, U)) > 0)
    
    # objective functions
    objective = objective_fn(X, U)
    opti.minimize(objective)
    
    if is_linear:
        opti.solver('osqp')
    else:
        opti.solver('ipopt')
    
    info["opti"] = opti
    
    # solve
    sol = opti.solve()    
    trajs_np, u_np = sol.value(X), sol.value(U)
    
    return trajs_np, u_np, info