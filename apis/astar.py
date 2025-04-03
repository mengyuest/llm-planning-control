import heapq

def solve_sequence(
    start_cell, 
    goal_cell,
    constraint_fn,
    heuristic_fn,
    choices=[(0, 1), (0,-1), (1,0), (-1,0)],
):
    info = {}
    open_set = []
    heapq.heappush(open_set, (0, start_cell))  # (priority, node)
    g_cost = {start_cell: 0}
    parent = {start_cell: (None, None)}
    while open_set:
        _, current = heapq.heappop(open_set)
        if current[0]==goal_cell[0] and current[1]==goal_cell[1]:
            trajs = []
            moves = []
            while current:
                trajs.append(current)
                current, move = parent[current]
                moves.append(move)
            trajs = trajs[::-1]
            moves = moves[::-1]
            return trajs, moves, info

        for move in choices:
            neighbor = (current[0]+move[0], current[1]+move[1])
            if constraint_fn(current, neighbor):
                continue
            new_g = g_cost[current] + 1
            if neighbor not in g_cost or new_g < g_cost[neighbor]:
                g_cost[neighbor] = new_g
                priority = new_g + heuristic_fn(neighbor)
                heapq.heappush(open_set, (priority, neighbor))
                parent[neighbor] = current, move
    return None, None, info