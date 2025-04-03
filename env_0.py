import random
import numpy as np
import torch
import casadi as ca
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from grid_env import GridEnv
import utils
import traceback

def sample_point(map_bounds):
    return np.array([np.random.uniform(low, high) for low, high in map_bounds])

def dist(p1, p2):
    return np.linalg.norm(p1 - p2)

def cast_list(sols):
    if len(sols) == 1:
        sols = sols[0]
    
    # so now either array([]) or [[(x,y),t]...]
    if isinstance(sols[0], list):
        if isinstance(sols[0][0], tuple) or isinstance(sols[0][0], list):
            # Handle [[((x0,y0),t0),...]] or [((x0,y0),t0),...]
            sols = np.array([[p[0][0], p[0][1], p[1]] for p in sols])
        else:
            # Handle [[[x0,y0,t0],...]] or [[x0,y0,t0],...]
            sols = np.array(sols)
    else:
        # Handle numpy array input
        sols = np.array(sols)
    return sols

class ControlEnv():
    def __init__(self, dynamics_type, task_type, 
                seed, x_min, x_max, u_min, u_max,
                num_obstacles=5, obstacle_type="mixed",
                nt=20, dt=1.0,
                min_clearance=0.5,
                min_radius=0.5,
                max_radius=1.0,
                ):
        
        utils.seed_everything(seed)
        
        assert dynamics_type in ["single", "double", "unicycle", "pendulum", "arm"]
        self.dynamics_type = dynamics_type
        if self.dynamics_type == "single":
            self.dynamics = self.single_dynamics
            self.dynamics_torch = self.single_dynamics_torch
            self.dynamics_casadi = self.single_dynamics_casadi
            self.dynamics_linearized_AB = self.single_dynamics_linearized_AB
        elif self.dynamics_type == "double":
            self.dynamics = self.double_dynamics
            self.dynamics_torch = self.double_dynamics_torch
            self.dynamics_casadi = self.double_dynamics_casadi
            self.dynamics_linearized_AB = self.double_dynamics_linearized_AB
        elif self.dynamics_type == "unicycle":
            self.dynamics = self.unicycle_dynamics
            self.dynamics_torch = self.unicycle_dynamics_torch
            self.dynamics_casadi = self.unicycle_dynamics_casadi
            self.dynamics_linearized_AB = self.unicycle_dynamics_linearized_AB
        
        assert obstacle_type in ["square", "circle", "mixed"]
        self.obstacle_type = obstacle_type
        
        assert task_type in [0, 1, 2, 3, 4, 5]
        self.task_type = task_type
        self.ref_points = None
        self.u_min = u_min
        self.u_max = u_max
        self.u_min_torch = torch.from_numpy(self.u_min).float()
        self.u_max_torch = torch.from_numpy(self.u_max).float()
        
        self.dt=dt
        self.nt=nt
        self.x_min = x_min
        self.x_max = x_max
        map_bounds = np.stack([x_min, x_max],axis=-1)
        self.x_dim = x_dim = x_min.shape[0]
        self.u_dim = u_dim = u_min.shape[0]        
        
        self.grid_env = None
        self.t_start = None
        self.t_end = None
        self.ref_traj = None
        self.ref_us =  None

        if self.task_type==3:  # Hiearachy planning (not control)
            assert x_dim==2
            assert obstacle_type=="mixed"
            self.grid_env = GridEnv(3, 3, n_lin=6, num_obstacles=num_obstacles, min_radius=0.2, max_radius=0.35)#, d_min=0.5, d_max=0.5)
            self.grid_size = self.grid_env.grid_size
            init = self.grid_env.start
            goal = self.grid_env.goal
            obstacles = self.grid_env.all_obstacles
            
        elif self.task_type in [4, 5]:  # STL
            init = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5)]).round(decimals=2)
            goal = None
            self.obj_A = [1, 9, 1.0, 1.0]
            self.obj_B = [2, 4.5, 4, 6]
            self.obj_C = [7.5, 6.0, 1.0, 8]
            self.obj_D = [9.0, 5.5, 2.0, 0.5]
            self.obj_E = [9.0, 8.2, 0.8, 0.8]
            
            flip = np.random.rand()>0.5
            if flip:
                init[0] = (self.x_max[0]+self.x_min[0])-init[0]
                self.obj_A[0] = (self.x_max[0]+self.x_min[0])-self.obj_A[0]
                self.obj_B[0] = (self.x_max[0]+self.x_min[0])-self.obj_B[0]
                self.obj_C[0] = (self.x_max[0]+self.x_min[0])-self.obj_C[0]
                self.obj_D[0] = (self.x_max[0]+self.x_min[0])-self.obj_D[0]
                self.obj_E[0] = (self.x_max[0]+self.x_min[0])-self.obj_E[0]
            
            # scaling
            scale_a = np.random.uniform(0.9, 1.1)
            self.obj_A[2] = (self.obj_A[2] * scale_a)
            self.obj_A[3] = (self.obj_A[3] * scale_a)
            
            self.obj_B[3] = self.obj_B[3] * np.random.uniform(0.8, 1.0)
            
            scale_e = np.random.uniform(0.9, 1.1)
            down_offset = np.random.uniform(-0.2, 0.8)
            self.obj_E[2] = (self.obj_E[2] * scale_e)
            self.obj_E[3] = (self.obj_E[3] * scale_e)
            self.obj_E[1] = self.obj_E[1] - down_offset
            self.obj_D[1] = self.obj_D[1] - down_offset * np.random.uniform(0.8, 1.2)
            
            obj_list=[self.obj_A, self.obj_B, self.obj_C, self.obj_D, self.obj_E]
            obstacles = [(2, obj[0], obj[1], obj[2], obj[3]) for obj in obj_list]
            self.t_start = np.random.randint(nt//2, nt-4)
            self.t_end = nt-3
            
        elif self.task_type==0:  # Tracking
            if self.dynamics_type=="unicycle":
                init = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5), 0, 0.]).round(decimals=2)
            else:
                init = init = np.array([np.random.uniform(0.5, 1.5), np.random.uniform(0.5, 1.5)]).round(decimals=2)
            goal = None
            obstacles = []
            
        else:  # path-planning
            # Sample init and goal, far enough
            map_diag = np.linalg.norm(map_bounds[:,1] - map_bounds[:,0])
            while True:
                init = sample_point(map_bounds)
                goal = sample_point(map_bounds)
                if dist(init, goal) >= 0.75 * map_diag:
                    break
            obstacles = []
            attempts = 0
            max_attempts = 10000
            while len(obstacles) < num_obstacles and attempts < max_attempts:
                if obstacle_type in ["circle", "square"]:
                    this_obstacle_type = obstacle_type
                else:
                    this_obstacle_type = "circle" if np.random.rand()<0.5 else "square"
                center = sample_point(map_bounds)
                radius = np.random.uniform(min_radius, max_radius)
                if dist(center, init) > (radius + min_clearance) and dist(center, goal) > (radius + min_clearance):
                    overlap = False
                    for _, cx, cy, r2 in obstacles:
                        c2 = np.array([cx, cy])
                        if dist(center, c2) < (radius + r2 + min_clearance):
                            overlap = True
                            break
                    if not overlap:
                        if this_obstacle_type=="circle":
                            obstacles.append((0, center[0], center[1], radius))
                        elif this_obstacle_type=="square":
                            obstacles.append((1, center[0], center[1], radius))
                        else:
                            raise NotImplementedError
                attempts += 1
        
        self.init = init
        self.goal = goal
        self.obstacles = obstacles
        
        if self.dynamics_type == "single":
            self.ego_state = np.array(init)
        elif self.dynamics_type == "double":
            self.ego_state = np.zeros(4)
            self.ego_state[:2] = init
        elif self.dynamics_type == "unicycle":
            self.ego_state = np.zeros(4)
            self.ego_state[:2] = init[:2]
        
        if self.task_type==0:    # tracking case
            self.ref_state = np.array(self.ego_state)
            if self.dynamics_type=="unicycle":
                self.ref_state[0] += np.random.uniform(0.1, 0.4) # 0.5
                self.ref_state[1] += np.random.uniform(0.5, 1.0) # 0.8
            else:
                self.ref_state[0] += np.random.uniform(0.2, 0.5) # 0.5
                self.ref_state[1] += np.random.uniform(1.5, 2.5) # 0.8
            self.ref_traj = [self.ref_state]
            self.ref_us = []
            test_flat = np.random.rand()>0.5
            if test_flat:
                ref_theta = np.random.uniform(-0.1, 0.3)
            else:
                ref_theta = 0            
            ref_omega = 0
            
            for _ in range(nt):
                x = np.array(self.ref_traj[-1])
                ref_theta += ref_omega
                if test_flat:
                    ref_omega = 0.0
                else:
                    ref_omega = np.random.uniform(-0.2, 0.3)
                ref_v = np.random.uniform(0.25, 0.4)
                x[0] = x[0] + ref_v * np.cos(ref_theta)
                x[1] = x[1] + ref_v * np.sin(ref_theta)
                if self.dynamics_type=="unicycle":
                    x[2] = ref_theta
                    x[3] = ref_v
                self.ref_traj.append(x)
            self.ref_traj = np.stack(self.ref_traj, axis=0)
        self.description = self.get_description()
    
    def get_dynamics_description(self, no_dynamics=False, timed=False):
        dynamics_str = ""
        if no_dynamics:
            if timed:
                dynamics_str += "Since this is temporal path planning, don't need to consider dynamics here. Just plan timed 2D waypoints in List((xk,yk),tk)."
            else:
                dynamics_str += "Since this is path planning, don't need to consider dynamics here. Just plan 2D waypoints."
        else:
            if self.dynamics_type=="single":
                dynamics_str += "The dynamics is single integrator. "
                dynamics_str += "The state (x, y) and control (vx, vy) satisfy "
                dynamics_str += "x_{t+1}=x_t+vx_t*dt, y_{t+1}=y_t+vy_t*dt."
            elif self.dynamics_type=="double":
                dynamics_str += "The dynamics is double integrator. "
                dynamics_str += "The state (x, y, vx, vy) and control (ax, ay) satisfy "
                dynamics_str += "x_{t+1}=x_t+vx_t*dt, y_{t+1}=y_t+vy_t*dt, vx_{t+1}=vx_t+ax_t*dt,  vy_{t+1}=vy_t+ay_t*dt."
            elif self.dynamics_type=="unicycle":
                dynamics_str += "The dynamics is unicycle model. "
                dynamics_str += "The state (x, y, theta, v) and control (omega, a) satisfy "
                dynamics_str += "x_{t+1} = x_t + v_t * cos(theta_t) * dt, "
                dynamics_str += "y_{t+1} = y_t + v_t * sin(theta_t) * dt, "
                dynamics_str += "theta_{t+1} = theta_t + omega_t * dt, "
                dynamics_str += "v_{t+1} = v_t + a_t * dt."
        dynamics_str += " "
        return dynamics_str
    
    def get_nt_dt_control_description(self):
        nt_dt_u_desc = ""
        nt_dt_u_desc += f"The control horizon is {self.nt} steps, and the time duration dt={self.dt:.4f}. "
        nt_dt_u_desc += f"The range for the control u is {self.u_min} <= u <= {self.u_max}. "
        return nt_dt_u_desc
    
    def str_list_vec(self, list_vec, prec="%.3f", split=",", n_cap=1000, v_cap=1000, flatten=False, first_int=False):
        if isinstance(list_vec, np.ndarray) and len(list_vec.shape)==1:
            return split.join(["%s"%(",".join([prec%vec])) for vec in list_vec[:v_cap]])
        else:
            if first_int:
                return split.join(["(%s)"%(",".join(["%d"%vec[0]]+[prec%x for x in vec[1:n_cap]])) for vec in list_vec[:v_cap]])
            else:
                return split.join(["(%s)"%(",".join([prec%x for x in vec[:n_cap]])) for vec in list_vec[:v_cap]])
    
    def get_description(self, LLM_MODE=None):
        if self.task_type==3:
            init_desc = "You are at point (%.3f, %.3f) in cell (%d,%d). "%(self.init[0], self.init[1], self.grid_env.init_cell[0], self.grid_env.init_cell[1])
        else:
            if self.dynamics_type=="unicycle":
                init_desc = "You are at point (%.3f, %.3f, %.3f, %.3f). "%(self.init[0], self.init[1], self.init[2], self.init[3])
            else:
                init_desc = "You are at point (%.3f, %.3f). "%(self.init[0], self.init[1])
        dynamics_desc = self.get_dynamics_description()
        nt_dt_u_desc = self.get_nt_dt_control_description()
        obstacles_desc = "The obstacles are in format List((type, x, y, radius)), with type=0 for circle and type=1 for square. For squares, side=sqrt(2)*radius. "
        obstacles_desc += "The obstacles are %s. "%(self.str_list_vec(self.obstacles, prec="%.3f", first_int=True))
        obstacles_desc += "You can access the obstacles variable using `env.obstacles`. "
        map_desc = "The trajectory cannot be outside of the map range: %.2f<=x<=%.2f, %.2f<=y<=%.2f. "%(self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1])
                
        task_description = ""
        if self.task_type==0:  # tracking reference path
            task_description += "Try to solve the following tracking problem. "
            task_description += init_desc
            if self.dynamics_type=="unicycle":
                task_description += "The goal is to track the reference path in the format of List((x, y, theta, v)) "
                task_description += "which is %s. "%(self.str_list_vec(self.ref_traj, prec="%.3f", n_cap=4))
            else:
                task_description += "The goal is to track the reference path in the format of List((x, y)) "
                task_description += "which is %s. "%(self.str_list_vec(self.ref_traj, prec="%.3f", n_cap=2))
            task_description += map_desc
            task_description += dynamics_desc
            task_description += nt_dt_u_desc
            task_description += "The control output solution should be in shape (%d,%d) to satisfy the spec.\n"%(self.nt, self.u_dim)
        
        elif self.task_type in [1, 2]: # 1- path planning for obstacles / 2- planning + tracking control
            if self.task_type==1:
                task_description += "Try to solve the following path planning problem. "
            else:
                task_description += "Try to solve the following planning + control problem. "
            task_description += init_desc
            task_description += "The goal is to reach the point (%.3f, %.3f) "%(self.goal[0], self.goal[1])
            task_description += "while avoiding obstacles (for both waypoints and paths) and keep the path in map range. "
            task_description += obstacles_desc
            task_description += map_desc
            
            if self.task_type==1:
                task_description += self.get_dynamics_description(no_dynamics=True)
            if self.task_type==2:
                task_description += dynamics_desc
                task_description += nt_dt_u_desc
                task_description += "The control output solution should be in shape (%d,%d) to satisfy the spec.\n"%(self.nt, self.u_dim)
            
        elif self.task_type==3: # Astar + RRT
            entry_points = self.grid_env.get_entry_points()
            task_description += "Try to solve the following hierarchical path-planning (global planning, local planning) problem. "
            task_description += "The hierarchical 2D env is %dx%d grid cells, where each cell contains 2D square or sphere obstacles. "%(self.grid_env.M, self.grid_env.N)
            task_description += "The row_i, col_j cell coordinates for the 2D point(x, y) is: row_i=int(y/grid_size), and col_j=int(x/grid_size)"
            task_description += "Each cell (row_i, col_j) spans col_j*grid_size <=x<= (col_j+1)*grid_size and row_i*grid_size<=y<=(row_i+1)*grid_size"
            task_description += init_desc
            task_description += "The goal is to reach the point (%.3f, %.3f) at cell (%d, %d) "%(self.goal[0], self.goal[1], self.grid_env.goal_cell[0], self.grid_env.goal_cell[1])
            task_description += "while avoiding obstacles (for both waypoints and paths) and keep the path in map range. "
            task_description += obstacles_desc
            task_description += map_desc
            task_description += "Cells have traverse constraints due to the boundaries and "
            task_description += "adjacent cells might have connections for traverse (in both direction) by the entry/exit point on their shared edges.\n"           
            for cell1, cell2, entry in entry_points:
                task_description += "Cells (%d, %d) connects (%d, %d) by point (%.3f %.3f).\n"%(cell1[0], cell1[1], cell2[0], cell2[1], entry[0], entry[1])
            task_description += "If generate code, don't use 'Define other transitions/obstacles similarly...' - write full information needed, because the code needs to be executable to solve this problem."
            task_description += "Hint: Step-1 Globally plan traversal on grid cells. "
            task_description += "Step-2 For each traversal, extract the entry point. Output: a list of waypoints = [init] + entry points + [goal]. "
            task_description += "Step-3 For consecutive waypoints, plan a local safe path within the corresponding cell. "
            task_description += "Step-4 Concatenate all paths segments so it will generate path from init to goal (dont need further tracking control). "
        
        elif self.task_type in [4, 5]: # signal temporal logic (4, planning; 5, control)
            M = {"A": self.obj_A, "B": self.obj_B, "C": self.obj_C, "D": self.obj_D, "E": self.obj_E}
            if self.task_type==4:
                task_description += "Try to solve the following temporal path planning problem (dont use tracking control). "
            else:
                task_description += "Try to solve the following temporal planning + control problem. "
            task_description += init_desc
            task_description += "The goal is to satisfy the Signal Temporal Logic (STL) constraints. "
            task_description += "The STL spec is ``eventually reach E during time range [%d, %d], "%(self.t_start, self.t_end)
            task_description += "and always avoid B and C from [%d, %d], and do not reach D, until first reach A from [%d, %d].`` "%(0, self.nt, 0, self.nt)
            task_description += "Here A, B, C, D, E are rectangle regions. "
            task_description += "Their center x,y, width, and height are %s. "%(
                ",".join(["%s:(%.3f,%.3f,%.3f,%.3f)"%(ele, M[ele][0], M[ele][1], M[ele][2], M[ele][3]) for ele in ["A", "B", "C", "D", "E"]]))
            task_description += map_desc
            if self.task_type==4:
                task_description += self.get_dynamics_description(no_dynamics=True, timed=True)
            if self.task_type==5:
                task_description += dynamics_desc
                task_description += nt_dt_u_desc
                task_description += "The control output solution should be in shape (%d,%d) to satisfy the spec.\n"%(self.nt, self.u_dim)
        else:
            raise NotImplementedError

        return task_description
    
    def get_env_api_description(self):
        env_api_description = "- system dynamics: `env.dynamics(x, u, dt)` for Numpy I/O (state x, control u, and time dt; outputs next_x), "
        env_api_description += "`env.dynamics_torch(x, u, dt)` for PyTorch Tensor I/O, `env.dynamics_casadi(x, u, dt)` for CasADi I/O, "
        env_api_description += "`dynamics_linearized_AB(self, x_ref, u_ref, dt)` returns linearized system matrices A and B at reference x_ref and u_ref.\n"
        env_api_description += "- environment: `env.init`: initial point, `env.goal`: goal point (if any), `env.obstacles` for obstacles in format List((type=0/1/2, x, y, radius, ...)), "
        env_api_description += "where type=0 means circle, type=1 for square (radius=side_length/sqrt(2)), and type=2 for rectangle (2, x, y, width, height). "
        env_api_description += "Whenever possible, use these apis rather than the values in prompt to access to the map information (init, goal, obstacles). "
        return env_api_description    
    
    def rollout(self, us):
        trajs = [self.ego_state]
        for ti in range(us.shape[-2]):
            new_x = self.dynamics(trajs[-1], us[..., ti, :], self.dt)
            trajs.append(new_x)
        trajs = np.stack(trajs, axis=-2)
        return trajs        
        
    def check_collision(self, trajs):
        # Check for collisions at nodes
        no_node_collision = True
        accident0_seg_i = -1
        accident0_obs_i = -1
        for i, point in enumerate(trajs):
            for obs_i,obs in enumerate(self.obstacles):
                obs_type, cx, cy, r = obs
                obs_center = np.array([cx, cy])
                if obs_type == 0:  # Circle
                    if np.linalg.norm(point[:2] - obs_center) <= r:
                        no_node_collision = False
                        accident0_seg_i = i
                        accident0_obs_i = obs_i
                        break
                elif obs_type == 1:  # Square
                    dx = abs(point[0] - cx)
                    dy = abs(point[1] - cy)
                    if dx <= r/np.sqrt(2) and dy <= r/np.sqrt(2):
                        no_node_collision = False
                        accident0_seg_i = i
                        accident0_obs_i = obs_i
                        break
            if not no_node_collision:
                break
                
        # Check for collisions along edges
        no_edge_collision = True
        accident1_seg_i = -1
        accident1_obs_i = -1
        for i in range(len(trajs)-1):
            start = trajs[i][:2]
            end = trajs[i+1][:2]
            # Sample points along segment
            interp_points = np.linspace(start, end, num=20)
            for obs_i, obs in enumerate(self.obstacles):
                obs_type, cx, cy, r = obs
                obs_center = np.array([cx, cy])
                if obs_type == 0:  # Circle
                    dists = np.linalg.norm(interp_points - obs_center[None,:], axis=1)
                    if np.any(dists <= r):
                        no_edge_collision = False
                        accident1_seg_i = i
                        accident1_obs_i = obs_i
                        break
                elif obs_type == 1:  # Square
                    dx = np.abs(interp_points[:,0] - cx)
                    dy = np.abs(interp_points[:,1] - cy)
                    if np.any((dx <= r/np.sqrt(2)) & (dy <= r/np.sqrt(2))):
                        no_edge_collision = False
                        accident1_seg_i = i
                        accident1_obs_i = obs_i
                        break
            if not no_edge_collision:
                break
        return no_node_collision, accident0_seg_i, accident0_obs_i, no_edge_collision, accident1_seg_i, accident1_obs_i
    
    
    def evaluate(self, sols):  # 0/1, score, runtime, others
        res_d = {"success": False, "prompt": "", "sols_raw": sols, "tracking_error":None}
        if self.task_type == 4:
            try:
                sols = cast_list(sols)  # Nx3 (x, y, t)
                assert sols.shape[0]>1 and sols.shape[1]==3
            except:
                res_d["prompt"] += "The solution cannot be casted. Maybe it is infeasible. Please check the format."
                sols = None
        
        if sols is None:
            res_d["prompt"] += "The solution is None. The algorithm does't return meaningful solution."
        else:
            try:
                sols = np.array(sols)                
                assert len(sols.shape)==2
                if sols.shape[0]<sols.shape[1]:
                    sols = sols.T
                res_d["sols"] = sols
                if self.task_type in [0, 2, 5]:
                    trajs = self.rollout(sols)
                else:
                    trajs = sols
                res_d["trajs"] = trajs
                assert len(res_d["trajs"].shape)==2 and res_d["trajs"].shape[0]>=2 and res_d["trajs"].shape[1]>=2
                if self.task_type==0:
                    trajs = trajs[:self.ref_traj.shape[0]]
                    full_error = np.linalg.norm(self.ref_traj-trajs, axis=-1)
            except:
                trajs = None
                res_d["trajs"] = None
                res_d["prompt"] += "The solution is not a valid 2D numpy array. Please check the format."

            try:
                if "trajs" in res_d and res_d["trajs"] is not None:
                    # checking success criteria
                    if self.task_type==0:
                        full_error = np.linalg.norm(self.ref_traj[:, :2]-trajs[:, :2], axis=-1)
                        error = np.mean(full_error)
                        if self.dynamics_type=="unicycle":
                            error_tol = 2.0
                        else:
                            error_tol = 1.0
                        res_d["tracking_error"] = error
                        if error < error_tol:
                            res_d["success"] = True
                        else:
                            res_d["prompt"] += "The simulated traj is %s. "%(self.str_list_vec(trajs, flatten=True))
                            res_d["prompt"] += "The reference traj is %s. "%(self.str_list_vec(self.ref_traj, flatten=True))
                            res_d["prompt"] += "The average tracking error is %.3f, greater than tracking tolerance:%.4f. "%(error, error_tol)
                            res_d["prompt"] += "The tracking error each time step is %s. "%(self.str_list_vec(full_error, flatten=True))
                    elif self.task_type in [1, 2, 3]:
                        # Check if endpoint is close to goal
                        goal_reach_tol = 1e-1
                        goal_reach_dist = np.linalg.norm(trajs[-1][:2] - self.goal[:2])
                        close_to_goal = goal_reach_dist <= goal_reach_tol
                        
                        # Check safety
                        no_node_collision, accident0_seg_i, accident0_obs_i, no_edge_collision, accident1_seg_i, accident1_obs_i = self.check_collision(trajs)
                        
                        if all([close_to_goal, no_node_collision, no_edge_collision]):
                            res_d["success"] = True
                        else:
                            res_d["prompt"] += "The simulated traj is %s. "%(self.str_list_vec(trajs, flatten=True))
                        if close_to_goal==False:
                            res_d["prompt"] += "The endpoint is %s, and the goal is %s, distance=%.3f>%.3f: too far from goal. "%(
                                trajs[-1], self.goal, goal_reach_dist, goal_reach_tol)
                        if no_node_collision==False:
                            res_d["prompt"] += "There exists a collision at step %d (%.3f, %.3f) with obstacle %s. "%(
                                accident0_seg_i, trajs[accident0_seg_i][0], trajs[accident0_seg_i][1], self.obstacles[accident0_obs_i])
                        if no_edge_collision==False:
                            res_d["prompt"] += "There exists a segment collision during steps %d (%.3f, %.3f) to %d (%.3f, %.3f)  with obstacle %s. "%(
                                accident1_seg_i, trajs[accident1_seg_i][0], trajs[accident1_seg_i][1], 
                                accident1_seg_i+1,  trajs[accident1_seg_i+1][0], trajs[accident1_seg_i+1][1], self.obstacles[accident1_obs_i])
                    elif self.task_type in [4, 5]:
                        # Check map bounds
                        map_bounds_violation = np.any(trajs[:,:self.x_dim] < self.x_min[:self.x_dim]) or np.any(trajs[:,:self.x_dim] > self.x_max[:self.x_dim])
                        if map_bounds_violation:
                            res_d["prompt"] += "The trajectory goes outside map bounds %.3f<=x<=%.3f, %.3f<=y<=%.3f. "%(self.x_min[0], self.x_max[0], self.x_min[1], self.x_max[1])
                            return res_d

                        # Define rectangular regions as (A,b) where Ax <= b
                        def make_rect_region(tmp_line):
                            center_x, center_y, width, height = tmp_line
                            A = np.array([[-1,0], [1,0], [0,-1], [0,1]])
                            b = np.array([
                                -center_x + width/2,
                                center_x + width/2,
                                -center_y + height/2,
                                center_y + height/2
                            ])
                            return (A,b)

                        region_A = make_rect_region(self.obj_A)
                        region_B = make_rect_region(self.obj_B)
                        region_C = make_rect_region(self.obj_C)
                        region_D = make_rect_region(self.obj_D)
                        region_E = make_rect_region(self.obj_E)

                        if self.task_type == 5:
                            # Augment trajectory with time dimension
                            times = np.arange(self.nt + 1) * self.dt
                            trajs = np.concatenate((trajs, times[:, None]), axis=-1)

                        # Densify trajectory for better checking
                        dense_trajs = []
                        dense_times = []
                        for i in range(len(trajs)-1):
                            p1, p2 = trajs[i,:2], trajs[i+1,:2]
                            t1, t2 = trajs[i,2], trajs[i+1,2]
                            num_points = max(int(np.linalg.norm(p2-p1)/0.1), 2)
                            for j in range(num_points):
                                alpha = j/num_points
                                p = p1*(1-alpha) + p2*alpha
                                t = t1*(1-alpha) + t2*alpha
                                dense_trajs.append(p)
                                dense_times.append(t)
                        dense_trajs.append(trajs[-1,:2])
                        dense_times.append(trajs[-1, 2])
                        dense_trajs = np.array(dense_trajs)
                        dense_times = np.array(dense_times)

                        def point_in_region(point, region):
                            A,b = region
                            return np.all(A @ point <= b)

                        # Check STL specifications
                        success = True
                        
                        # Eventually reach E during [12,17]
                        reach_E = False
                        for i in range(len(dense_trajs)):
                            if self.t_start <= dense_times[i] <= self.t_end:
                                if point_in_region(dense_trajs[i], region_E):
                                    reach_E = True
                                    break
                        if not reach_E:
                            success = False
                            res_d["prompt"] += "Failed to reach region E during time [%d,%d]. "%(self.t_start, self.t_end)

                        # Always avoid B and C during [0,20]
                        for i in range(len(dense_trajs)):
                            if dense_times[i] <= 20:
                                if point_in_region(dense_trajs[i], region_B):
                                    success = False
                                    res_d["prompt"] += f"Collided with region B at time {dense_times[i]:.2f}. "
                                if point_in_region(dense_trajs[i], region_C):
                                    success = False
                                    res_d["prompt"] += f"Collided with region C at time {dense_times[i]:.2f}. "
                            if success==False:
                                break
                        # Don't reach D until first reach A during [0,20]
                        reached_A = False
                        for i in range(len(dense_trajs)):
                            if dense_times[i] <= 20:
                                if point_in_region(dense_trajs[i], region_A):
                                    reached_A = True
                                if not reached_A and point_in_region(dense_trajs[i], region_D):
                                    success = False
                                    res_d["prompt"] += f"Reached region D before region A at time {dense_times[i]:.2f}. "
                            if success==False:
                                break
                        if success:
                            res_d["success"] = True
                        else:
                            res_d["prompt"] += "The simulated traj is %s. "%(self.str_list_vec(trajs, flatten=True))
                else:                
                    res_d["trajs"] = None
                    res_d["success"] = False
            except:
                res_d["trajs"] = None
                res_d["success"] = False
                res_d["prompt"] += "The evaluation failed - please check the format of the solution. "
                res_d["prompt"] += "The error is %s.\n"%(str(traceback.print_exc()))
                
        res_d["diagnose"] = res_d["prompt"]
        return res_d
    
    def visualization(self, trajs):
        marker_size = 64
        fontsize=18
        color_list = ["blue", "gray", "darkgray", "red", "green"]
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        for obs_i, obs in enumerate(self.obstacles):
            if obs[0]==0:
                circ = Circle(obs[1:3], radius=obs[3], color="gray")
                ax.add_patch(circ)
            elif obs[0]==1:
                a = obs[3]/np.sqrt(2)
                rect = Rectangle([obs[1]-a, obs[2]-a], a*2,a*2, color="gray")
                ax.add_patch(rect)
            elif obs[0]==2:
                w, h = obs[3], obs[4]
                rect = Rectangle([obs[1]-w/2, obs[2]-h/2], w, h, color=color_list[obs_i])
                ax.add_patch(rect)
            else:
                raise NotImplementedError
        
        plt.scatter(self.init[0], self.init[1], color="orange", s=marker_size, label="init", zorder=9999)
        if self.goal is not None:
            plt.scatter(self.goal[0], self.goal[1], color="green", s=marker_size, label="goal", zorder=9999)
        if self.ref_traj is not None:
            plt.plot(self.ref_traj[:, 0], self.ref_traj[:, 1], color="darkgray", linewidth=3.0, linestyle="--", label="ref_traj", zorder=99)
        if trajs is not None:
            plt.plot(trajs[:, 0], trajs[:, 1], color="royalblue", linewidth=3.0, label="traj", zorder=999)
        plt.axis("scaled")
        plt.legend(loc='upper right', ncol=4, fontsize=fontsize, bbox_to_anchor=(1.0, 1.0), framealpha=0.9,)
        plt.xlim(self.x_min[0], self.x_max[0])
        plt.ylim(self.x_min[1], self.x_max[1])
        return None

    # ************************************************************** #
    # single integrator
    def single_dynamics(self, x, u, dt):  # np.array version
        u = np.clip(u, self.u_min, self.u_max)
        delta_x = u * dt
        next_x = x + delta_x
        return next_x
    
    def single_dynamics_torch(self, x, u, dt):
        u = torch.clip(u, self.u_min_torch, self.u_max_torch)
        delta_x = u * dt
        next_x = x + delta_x
        return next_x
    
    def single_dynamics_casadi(self, x, u, dt):
        next_x = x + u * dt
        return next_x
    
    def single_dynamics_linearized_AB(self, x_ref, u_ref, dt):
        A, B = np.eye(2), np.eye(2)*dt
        return A, B
        
    # ************************************************************** #
    # unicycle model
    def unicycle_dynamics(self, x, u, dt):
        u = np.clip(u, self.u_min, self.u_max)
        return np.stack([
            x[..., 0] + x[..., 3] * np.cos(x[..., 2]) * dt,
            x[..., 1] + x[..., 3] * np.sin(x[..., 2]) * dt,
            x[..., 2] + u[..., 0] * dt,
            x[..., 3] + u[..., 1] * dt], axis=-1)

    def unicycle_dynamics_torch(self, x, u, dt):
        u = torch.clip(u, self.u_min, self.u_max)
        return torch.stack([
            x[..., 0] + x[..., 3] * torch.cos(x[..., 2]) * dt,
            x[..., 1] + x[..., 3] * torch.sin(x[..., 2]) * dt,
            x[..., 2] + u[..., 0] * dt,
            x[..., 3] + u[..., 1] * dt], dim=-1)
    

    def unicycle_dynamics_casadi(self, x, u, dt):
        return ca.vertcat(
            x[0] + x[3] * ca.cos(x[2]) * dt,
            x[1] + x[3] * ca.sin(x[2]) * dt,
            x[2] + u[0] * dt,
            x[3] + u[1] * dt,
        )
    
    def unicycle_dynamics_linearized_AB(self, x_ref, u_ref, dt):
        _, _, theta, v = x_ref

        A = np.eye(4)
        A[0, 2] = -v * np.sin(theta) * dt
        A[0, 3] = np.cos(theta) * dt
        A[1, 2] = v * np.cos(theta) * dt
        A[1, 3] = np.sin(theta) * dt

        B = np.zeros((4, 2))
        B[2, 0] = dt      # ∂θ_next / ∂ω
        B[3, 1] = dt      # ∂v_next / ∂a

        return A, B

    
    