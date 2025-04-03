import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import random
from collections import deque

class GridEnv:
    def __init__(self, M=3, N=3, x_min=0., x_max=10., y_min=0., y_max=10., n_lin=5, num_obstacles=5, min_radius=0.25, max_radius=0.5, connection_rate=0.85, bloat=0.15, d_min=0.1, d_max=0.9):
        self.M = M  # rows
        self.N = N  # columns
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.num_obstacles = num_obstacles
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.connection_rate = connection_rate
        
        self.n_lin = n_lin
        self.bloat=bloat
        
        self.grid_size = (x_max-x_min) / N  # size of each grid cell
        self.obstacles = {}  # obstacles in each grid cell
        self.all_obstacles = []
        self.connections = {}  # connections between adjacent cells
        self.start = None  # Will store (global_x, global_y)
        self.goal = None   # Will store (global_x, global_y)
        self.feasible_path = None
        
        self.d_min, self.d_max = d_min, d_max
        
        # Initialize environment
        self.generate_obstacles()
        self.generate_connections()
        self.generate_start_goal()
        self.all_obstacles = [obs for cell_obstacles in self.obstacles.values() for obs in cell_obstacles]
    
    def get_entry_points(self):
        """
        Get list of entry/exit points between connected cells.
        
        Returns:
            List of tuples (cell1, cell2, point) where:
            - cell1, cell2: Connected grid cell coordinates (i,j)
            - point: 2D Entry/exit point coordinate along shared boundary
        """
        entry_points = []
        for (cell1, cell2), point in self.connections.items():
            # Check if horizontal or vertical connection
            is_horizontal = cell1[0] == cell2[0]
            # Convert point to global xy coordinates
            if is_horizontal:
                # For horizontal connections, point is the y-coordinate in cell1's frame
                global_x = cell1[1]*self.grid_size + self.grid_size  # At boundary between cells
                global_y = cell1[0]*self.grid_size + point
            else:
                # For vertical connections, point is the x-coordinate in cell1's frame  
                global_x = cell1[1]*self.grid_size + point
                global_y = cell1[0]*self.grid_size + self.grid_size  # At boundary between cells
            entry_points.append((cell1, cell2, (global_x, global_y)))
            
        return entry_points

    def compute_cell(self, x, y):
        """Convert global coordinates to cell indices"""
        cell_i = int(y/self.grid_size)
        cell_j = int(x/self.grid_size)
        return (cell_i, cell_j)
        
    def is_point_in_obstacle(self, point, cell_obstacles):
        bloat=self.bloat
        x, y = point        
        for obs in cell_obstacles:
            center_x = obs[1]
            center_y = obs[2]
            if obs[0] == 0:
                dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if dist <= obs[3]+bloat:  # Within circle radius
                    return True
            else:  # square
                side = obs[3] * np.sqrt(2)
                if (x >= center_x-side/2-bloat and x <= center_x + side/2+bloat and 
                    y >= center_y-side/2-bloat and y <= center_y + side/2+bloat):
                    return True
        return False

    def do_obstacles_overlap(self, new_obs, existing_obstacles):
        bloat = self.bloat
        new_x, new_y = new_obs[1], new_obs[2]
        new_radius = new_obs[3]
        new_is_circle = new_obs[0] == 0

        for obs in existing_obstacles:
            x, y = obs[1], obs[2]
            radius = obs[3]
            is_circle = obs[0] == 0

            # Calculate distance between centers
            dist = np.sqrt((new_x - x)**2 + (new_y - y)**2)

            if new_is_circle and is_circle:
                # Circle-Circle collision
                if dist <= (new_radius + radius + bloat):
                    return True
            elif new_is_circle and not is_circle:
                # Circle-Square collision
                square_side = radius * np.sqrt(2)
                # Approximate circle-square collision with a slightly larger square
                if (abs(new_x - x) <= (square_side/2 + new_radius + bloat) and 
                    abs(new_y - y) <= (square_side/2 + new_radius + bloat)):
                    return True
            elif not new_is_circle and is_circle:
                # Square-Circle collision
                new_side = new_radius * np.sqrt(2)
                if (abs(new_x - x) <= (new_side/2 + radius + bloat) and 
                    abs(new_y - y) <= (new_side/2 + radius + bloat)):
                    return True
            else:
                # Square-Square collision
                new_side = new_radius * np.sqrt(2)
                side = radius * np.sqrt(2)
                if (abs(new_x - x) <= (new_side/2 + side/2 + bloat) and 
                    abs(new_y - y) <= (new_side/2 + side/2 + bloat)):
                    return True

        return False
        
    def generate_obstacles(self):
        margin = self.grid_size/self.n_lin/2
        
        # Generate random obstacles for each grid cell
        for i in range(self.M):
            for j in range(self.N):
                self.obstacles[(i,j)] = []
                num_obstacles = self.num_obstacles  # random.randint(3, 5)  # Increased number of obstacles
                
                # Ensure mix of circles and squares
                circle_count = random.randint(1, num_obstacles-1)
                square_count = num_obstacles - circle_count
                
                # Generate circles
                attempts = 0
                max_attempts = 100
                for _ in range(circle_count):
                    while attempts < max_attempts:
                        radius = random.uniform(self.min_radius, self.max_radius)
                        local_x = random.uniform(radius+margin, self.grid_size-radius-margin)
                        local_y = random.uniform(radius+margin, self.grid_size-radius-margin)
                        
                        # Convert to absolute coordinates
                        abs_x = j * self.grid_size + local_x
                        abs_y = i * self.grid_size + local_y
                        
                        new_obstacle = [0, abs_x, abs_y, radius]
                        if not self.do_obstacles_overlap(new_obstacle, self.obstacles[(i,j)]):
                            self.obstacles[(i,j)].append(new_obstacle)
                            break
                        attempts += 1
                
                # Generate squares
                attempts = 0
                for _ in range(square_count):
                    while attempts < max_attempts:
                        radius = random.uniform(self.min_radius, self.max_radius)
                        side = radius * np.sqrt(2)
                        local_x = random.uniform(side/2+margin, self.grid_size-side/2-margin)
                        local_y = random.uniform(side/2+margin, self.grid_size-side/2-margin)
                        
                        # Convert to absolute coordinates
                        abs_x = j * self.grid_size + local_x
                        abs_y = i * self.grid_size + local_y
                        
                        new_obstacle = [1, abs_x, abs_y, radius]
                        if not self.do_obstacles_overlap(new_obstacle, self.obstacles[(i,j)]):
                            self.obstacles[(i,j)].append(new_obstacle)
                            break
                        attempts += 1
                        
    def generate_connections(self):
        bloat=self.bloat
        max_attempts = 100
        n_lin = self.n_lin
        # Generate random connections between adjacent cells
        for i in range(self.M):
            for j in range(self.N):
                if i < self.M-1:  # vertical connection
                    if random.random() < self.connection_rate:  # 70% chance of connection
                        # Keep trying until we find a valid point
                        for _ in range(max_attempts):
                            point = random.uniform(self.d_min, self.d_max)*self.grid_size
                            # Check both cells for obstacles at connection point
                            if not (self.is_point_in_obstacle((j*self.grid_size + point, i*self.grid_size + self.grid_size), 
                                                            self.obstacles[(i,j)]) or
                                   self.is_point_in_obstacle((j*self.grid_size + point, i*self.grid_size), 
                                                            self.obstacles[(i+1,j)])):
                                self.connections[((i,j), (i+1,j))] = point
                                break
                    
                    # add boundary obstacles
                    xs = np.linspace(0, self.grid_size, n_lin+1, endpoint=True) + j * self.grid_size
                    side = self.grid_size/n_lin
                    for x in xs:
                        cube_x = x
                        cube_y = (i+1)*self.grid_size
                        collide = False
                        if ((i,j), (i+1,j)) in self.connections:
                            point = self.connections[((i,j), (i+1,j))]
                            real_px = j*self.grid_size + point
                            real_py = i*self.grid_size + self.grid_size
                            if cube_x-side/2-bloat<=real_px<=cube_x+side/2+bloat and cube_y-side/2-bloat<=real_py<=cube_y+side/2+bloat:
                                collide=True
                        if not collide:
                            self.obstacles[(i,j)].append([1, cube_x, cube_y, side/np.sqrt(2)])                
                        
                if j < self.N-1:  # horizontal connection
                    if random.random() < self.connection_rate:  # 70% chance of connection
                        # Keep trying until we find a valid point
                        for _ in range(max_attempts):
                            point = random.uniform(self.d_min, self.d_max)*self.grid_size
                            # Check both cells for obstacles at connection point
                            if not (self.is_point_in_obstacle((j*self.grid_size + self.grid_size, i*self.grid_size + point),
                                                            self.obstacles[(i,j)]) or
                                   self.is_point_in_obstacle((j*self.grid_size, i*self.grid_size + point),
                                                            self.obstacles[(i,j+1)])):
                                self.connections[((i,j), (i,j+1))] = point
                                break
                    
                    # add boundary obstacles
                    ys = np.linspace(0, self.grid_size, n_lin+1, endpoint=True) + i * self.grid_size
                    side = self.grid_size/n_lin
                    for y in ys:
                        cube_x = (j+1)*self.grid_size
                        cube_y = y
                        collide = False
                        if ((i,j), (i,j+1)) in self.connections:
                            point = self.connections[((i,j), (i,j+1))]
                            real_px = j*self.grid_size + self.grid_size
                            real_py = i*self.grid_size + point
                            if cube_x-side/2-bloat<=real_px<=cube_x+side/2+bloat and cube_y-side/2-bloat<=real_py<=cube_y+side/2+bloat:
                                collide=True
                        if not collide:
                            self.obstacles[(i,j)].append([1, cube_x, cube_y, side/np.sqrt(2)])       
    
    def find_path_exists(self, start_cell, goal_cell):
        # BFS to find path between cells
        visited = set()
        queue = deque([(start_cell, [start_cell])])
        visited.add(start_cell)
        
        while queue:
            current, path = queue.popleft()
            if current == goal_cell:
                return path
                
            # Check all possible connections
            for (cell1, cell2), _ in self.connections.items():
                if cell1 == current and cell2 not in visited:
                    queue.append((cell2, path + [cell2]))
                    visited.add(cell2)
                elif cell2 == current and cell1 not in visited:
                    queue.append((cell1, path + [cell1]))
                    visited.add(cell1)
                    
        return None
       
    def generate_start_goal(self):
        # Randomly select start and goal cells
        cells = [(i,j) for i in range(self.M) for j in range(self.N)]
        max_cell_attempts = 10000
        cell_attempt = 0
        
        while cell_attempt < max_cell_attempts:
            start_cell = random.choice(cells)
            remaining_cells = cells.copy()
            remaining_cells.remove(start_cell)
            goal_cell = random.choice(remaining_cells)
            
            # Check if path exists between cells
            if cell_attempt<max_cell_attempts//2:
                path = self.find_path_exists(start_cell, goal_cell)
                path = path if path is not None and len(path)>=(self.M+self.N-2)*0.8 else None
            else:
                path = self.find_path_exists(start_cell, goal_cell)
            if path:
                self.feasible_path = path
                break
            cell_attempt += 1
        
        if cell_attempt >= max_cell_attempts:
            raise Exception("Could not find valid start and goal cells with connecting path")
        
        self.init_cell = start_cell
        self.goal_cell = goal_cell
        
        # Keep trying until we find valid positions with proper clearance
        max_pos_attempts = 10000
        
        # Generate start position
        for _ in range(max_pos_attempts):
            local_start_x = random.uniform(0.25, 0.75)*self.grid_size
            local_start_y = random.uniform(0.25, 0.75)*self.grid_size
            global_start_x = start_cell[1]*self.grid_size + local_start_x
            global_start_y = start_cell[0]*self.grid_size + local_start_y
            if not self.is_point_in_obstacle((global_start_x, global_start_y), self.obstacles[start_cell]):
                self.start = (global_start_x, global_start_y)
                break
        else:
            raise Exception("Could not find valid start position")
                
        # Generate goal position
        for _ in range(max_pos_attempts):
            local_goal_x = random.uniform(0.25, 0.75)*self.grid_size
            local_goal_y = random.uniform(0.25, 0.75)*self.grid_size
            global_goal_x = goal_cell[1]*self.grid_size + local_goal_x
            global_goal_y = goal_cell[0]*self.grid_size + local_goal_y
            if not self.is_point_in_obstacle((global_goal_x, global_goal_y), self.obstacles[goal_cell]):
                self.goal = (global_goal_x, global_goal_y)
                break
        else:
            raise Exception("Could not find valid goal position")

        self.init = self.start
        self.goal = self.goal
        
        
    def evaluate(self, plan_path):
        """
        Evaluate if a planned path is valid:
        1. Starts at start position
        2. Ends at goal position 
        3. Does not collide with obstacles
        
        Args:
            plan_path: List of tuples (x,y) representing path points in global coordinates
            
        Returns:
            bool: True if path is valid, False otherwise
        """
        if not plan_path:
            return False
            
        # Check start and end points with threshold
        epsilon = 0.05
        start_pos = plan_path[0]
        goal_pos = plan_path[-1]
            
        # Check positions within epsilon
        start_diff = np.sqrt((start_pos[0] - self.start[0])**2 + 
                           (start_pos[1] - self.start[1])**2)
        goal_diff = np.sqrt((goal_pos[0] - self.goal[0])**2 + 
                          (goal_pos[1] - self.goal[1])**2)
        
        if start_diff > epsilon or goal_diff > epsilon:
            return False
            
        # Check each point along path for collisions
        for pos in plan_path:
            cell = self.compute_cell(pos[0], pos[1])
            local_x = pos[0] - cell[1]*self.grid_size
            local_y = pos[1] - cell[0]*self.grid_size
            if self.is_point_in_obstacle((local_x, local_y), self.obstacles[cell]):
                return False
                
        # Check each path segment for collisions
        for i in range(len(plan_path)-1):
            pos1 = plan_path[i]
            pos2 = plan_path[i+1]
            
            # Points are already in global coordinates
            x1, y1 = pos1
            x2, y2 = pos2
            
            # Check points along line segment
            for t in np.linspace(0, 1, 20):
                x = x1 + t*(x2-x1)
                y = y1 + t*(y2-y1)
                
                cell = self.compute_cell(x, y)
                pos_x = x - cell[1]*self.grid_size
                pos_y = y - cell[0]*self.grid_size
                
                if self.is_point_in_obstacle((pos_x,pos_y), self.obstacles[cell]):
                    return False
                    
        return True
        
    def plot_environment(self, path=None):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot grid cells and boundary walls
        for i in range(self.M):
            for j in range(self.N):
                # Plot cell outline
                ax.add_patch(Rectangle((j*self.grid_size, i*self.grid_size),
                                     self.grid_size, self.grid_size,
                                     fill=False, color='black'))
                
                # Plot obstacles
                for obs in self.obstacles[(i,j)]:
                    if obs[0] == 0:
                        ax.add_patch(Circle((obs[1], obs[2]),
                                          obs[3], color='gray', alpha=0.5))
                    else:  # square
                        ax.add_patch(Rectangle((obs[1]-obs[3]*np.sqrt(2)/2,
                                             obs[2]-obs[3]*np.sqrt(2)/2),
                                             obs[3]*np.sqrt(2), obs[3]*np.sqrt(2),
                                             color='gray', alpha=0.5))
                        
        # Plot connections
        for (cell1, cell2), point in self.connections.items():
            if cell1[0] == cell2[0]:  # horizontal connection
                i, j = cell1
                ax.plot([j*self.grid_size + self.grid_size, j*self.grid_size + self.grid_size],
                       [i*self.grid_size + point, i*self.grid_size + point],
                       'g-', linewidth=2)
                # Plot entry points
                ax.plot(j*self.grid_size + self.grid_size, i*self.grid_size + point,
                       'yo', markersize=6)
                ax.plot(j*self.grid_size + self.grid_size, i*self.grid_size + point,
                       'ko', markersize=3)
            else:  # vertical connection
                i, j = cell1
                ax.plot([j*self.grid_size + point, j*self.grid_size + point],
                       [i*self.grid_size + self.grid_size, i*self.grid_size + self.grid_size],
                       'g-', linewidth=2)
                # Plot entry points
                ax.plot(j*self.grid_size + point, i*self.grid_size + self.grid_size,
                       'yo', markersize=6)
                ax.plot(j*self.grid_size + point, i*self.grid_size + self.grid_size,
                       'ko', markersize=3)
                
        # Plot start and goal (already in global coordinates)
        ax.plot(self.start[0], self.start[1], 'bo', markersize=10, label='Start')
        ax.plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        # Plot feasible path through grid cells
        if self.feasible_path:
            print("feasible", self.feasible_path)
            print("connect",self.connections)
            path_points = []
            # Add start point
            path_points.append(self.start)  # Use global coordinates
            
            # Add connection points between cells
            for i in range(len(self.feasible_path)-1):
                cell1 = self.feasible_path[i]
                cell2 = self.feasible_path[i+1]
                if ((cell1, cell2) in self.connections):
                    point = self.connections[(cell1, cell2)]
                    if cell1[0] == cell2[0]:  # horizontal connection
                        path_points.append((cell1[1]*self.grid_size + self.grid_size,
                                         cell1[0]*self.grid_size + point))
                    else:  # vertical connection
                        path_points.append((cell1[1]*self.grid_size + point,
                                         cell1[0]*self.grid_size + self.grid_size))
                elif ((cell2, cell1) in self.connections):
                    point = self.connections[(cell2, cell1)]
                    if cell1[0] == cell2[0]:  # horizontal connection
                        path_points.append((cell2[1]*self.grid_size + self.grid_size,
                                         cell2[0]*self.grid_size + point))
                    else:  # vertical connection
                        path_points.append((cell2[1]*self.grid_size + point,
                                         cell2[0]*self.grid_size + self.grid_size))
            
            # Add goal point
            path_points.append(self.goal)  # Use global coordinates
            
            # Plot the path
            path_x = [p[0] for p in path_points]
            path_y = [p[1] for p in path_points]
            ax.plot(path_x, path_y, 'b--', linewidth=2, label='Feasible Path')
            
        # Plot custom path if provided
        if path is not None:
            path_x = [pos[0] for pos in path]  # Use global coordinates directly
            path_y = [pos[1] for pos in path]
            ax.plot(path_x, path_y, 'g--', linewidth=2, label='Custom Path')
            
        ax.set_xlim(-1, self.N*self.grid_size + 1)
        ax.set_ylim(-1, self.M*self.grid_size + 1)
        ax.legend()
        plt.axis('equal')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    random.seed(1007)
    np.random.seed(1007)
    env = GridEnv(6, 6)
    
    # Example path (you would normally compute this using a path planning algorithm)
    example_path = [
        env.start,  # start (global position)
        (1*env.grid_size + env.grid_size/2, 1*env.grid_size + env.grid_size/2),  # middle of intermediate cell
        env.goal  # goal (global position)
    ]
    
    # Evaluate and plot the path
    is_valid = env.evaluate(example_path)
    print(f"Path is valid: {is_valid}")
    env.plot_environment(path=example_path)
