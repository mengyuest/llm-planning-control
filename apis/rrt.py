import numpy as np

class Node:
    def __init__(self, pos, parent=None):
        self.position = np.array(pos)
        self.parent = parent
        self.cost = 0.0

def nearest_neighbor(nodes, sample, norm_fn):
    return min(nodes, key=lambda n: norm_fn(sample, n.position))

def find_neighbors(new_node, nodes, radius, norm_fn):
    return [n for n in nodes if norm_fn(new_node.position, n.position) <= radius]

def choose_parent(new_node, near_nodes):
    if not near_nodes: return new_node
    best = min(near_nodes, key=lambda n: n.cost + np.linalg.norm(n.position - new_node.position))
    new_node.parent = best
    new_node.cost = best.cost + np.linalg.norm(best.position - new_node.position)
    return new_node

def steer(from_node, to_point, step_size, norm_fn):
    d = to_point - from_node.position
    dist = norm_fn(to_point, from_node.position)
    pos = from_node.position + (d / dist) * step_size if dist > step_size else to_point
    return Node(pos, parent=from_node)

def rewire(new_node, neighbors, norm_fn, edge_fn):
    for n in neighbors:
        cost = new_node.cost + norm_fn(n.position, new_node.position)
        if n.cost > cost and not edge_fn(n.position, new_node.position):
            n.parent = new_node
            n.cost = cost

def extract_path(goal):
    path = []
    while goal:
        path.append(goal.position)
        goal = goal.parent
    return path[::-1]

def solve_sequence(start, 
                   goal, 
                   bounds,
                   config_collision_fn, 
                   edge_collision_fn,
                   step_size=0.5,
                   max_iter=1000,
                   goal_sample_rate=0.1,
                   use_rrt_star=False):
    def config_norm_fn(x, y):
        return np.linalg.norm(x-y, ord=2)

    init_goal = goal
    def goal_sample_fn():
        return init_goal
    
    def config_sample_fn():
        return np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    
    start, goal = Node(start), Node(goal)
    nodes = [start]
    info = {"nodes": nodes}

    for _ in range(max_iter):
        sample = goal_sample_fn() if np.random.rand() < goal_sample_rate else config_sample_fn()
        nearest = nearest_neighbor(nodes, sample, config_norm_fn)
        new_node = steer(nearest, sample, step_size, config_norm_fn)
        if config_collision_fn(new_node.position): continue

        if use_rrt_star:
            near = find_neighbors(new_node, nodes, 2 * step_size, config_norm_fn)
            new_node = choose_parent(new_node, near)
            if edge_collision_fn(new_node.parent.position, new_node.position): continue
            nodes.append(new_node)
            rewire(new_node, near, config_norm_fn, edge_collision_fn)
        else:
            if edge_collision_fn(new_node.parent.position, new_node.position): continue
            nodes.append(new_node)

        if config_norm_fn(new_node.position, goal.position) < step_size:
            goal.parent = new_node
            nodes.append(goal)
            return extract_path(goal), None, info

    return None, None, info