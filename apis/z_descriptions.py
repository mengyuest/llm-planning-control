

description_d0 = {}
description_d0["astar"]="A* (ASTAR) search with user-defined logic."
description_d0["cem"]="Trajectory optimization using the Cross Entropy Method (CEM)"
description_d0["grad"]="Trajectory optimization via gradient descent (GRAD) using PyTorch and backpropagation through time."
description_d0["lqr"]="Path tracking control using Linear Quadratic Regulator (LQR)."
description_d0["milp"]="End-to-end timed path planning for Signal Temporal Logic (STL) goal and constraint specs via Mixed-Integer Linear Program (MILP)."
description_d0["mpc"]="Model predictive control (MPC) using CasADi, supporting both linear and nonlinear dynamics with constraints."
description_d0["pid"]="Runs a Proportional-Integral-Derivative (PID) controller to follow a sequence of waypoints using a dynamics model."
description_d0["rrt"]="Rapidly-exploring Random Tree (RRT) or RRT* planner to find a collision-free path from a start to a goal state."

def get_code_usage(filepath):
    buffer=[]
    lines = open(filepath).readlines()
    for line in lines:
        # Remove inline comments
        line = line.split('#')[0]
        # Skip empty lines and comment-only lines
        if len(line.strip()) <= 1:
            continue
        buffer.append(line)
    new_buffer = []
    
    start_to_write = False
    for li, line in enumerate(buffer):
        if "'''" in line:
            if start_to_write==False:
                start_to_write = True
            else:
                new_buffer.append(line)
                new_buffer.append(buffer[li+1])
                break
        if start_to_write: 
            new_buffer.append(line)   
    return "".join(new_buffer)

def get_code_lines(filepath):
    buffer=[]
    lines = open(filepath).readlines()
    for line in lines:
        # Remove inline comments
        line = line.split('#')[0]
        # Skip empty lines and comment-only lines
        if len(line.strip()) <= 1:
            continue
        buffer.append(line)    
    return "".join(buffer)
