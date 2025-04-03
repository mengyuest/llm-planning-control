from gurobipy import *
from math import *
import numpy as np
import time
M = 1e3
T_MIN_SEP = 1e-2
IntFeasTol  = 1e-1 * T_MIN_SEP / M
def setM(v):
    global M, IntFeasTol
    M = v
    IntFeasTol  = 1e-1 * T_MIN_SEP / M
EPS = 1e-2

def stl_until(phi1, phi2, t1, t2): return Node('U', deps=[phi1, phi2], info={'int': [t1, t2]})
def stl_and(*args): return Node('and', deps=list(args))
def stl_mu(A, b): return Node('mu', info={'A': A, 'b': b})
def stl_negmu(A, b): return Node('negmu', info={'A': A, 'b': b})
def stl_always(phi, t1, t2): return Node('A', deps=[phi], info={'int': [t1, t2]})
def stl_eventually(phi, t1, t2): return Node('F', deps=[phi], info={'int': [t1, t2]})

def inside(region): return stl_mu(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), np.array([-region[0]+region[2]/2, region[0]+region[2]/2, -region[1]+region[3]/2, region[1]+region[3]/2]))
def outside(region): return stl_negmu(np.array([[-1, 0], [1, 0], [0, -1], [0, 1]]), np.array([-region[0]+region[2]/2, region[0]+region[2]/2, -region[1]+region[3]/2, region[1]+region[3]/2]))
def reach(region, t1=0, t2=10): return stl_eventually(inside(region), t1, t2)
def avoid(region, t1=0, t2=10): return stl_always(outside(region), t1, t2)
def stay(region, t1=0, t2=10): return stl_always(inside(region), t1, t2)

def _sub(x1, x2):
    return [x1[i] - x2[i] for i in range(len(x1))]
def L1Norm(model, x):
    xvar = model.addVars(len(x), lb=-GRB.INFINITY)
    abs_x = model.addVars(len(x))
    model.update()
    xvar = [xvar[i] for i in range(len(xvar))]
    abs_x = [abs_x[i] for i in range(len(abs_x))]
    for i in range(len(x)):
        model.addConstr(xvar[i] == x[i])
        model.addConstr(abs_x[i] == abs_(xvar[i]))
    return sum(abs_x)
class Conjunction(object):
    def __init__(self, deps = []):
        super(Conjunction, self).__init__()
        self.deps = deps
        self.constraints = []
class Disjunction(object):
    def __init__(self, deps = []):
        super(Disjunction, self).__init__()
        self.deps = deps
        self.constraints = []
def noIntersection(a, b, c, d):
    return Disjunction([c-b-EPS, a-d-EPS])
def hasIntersection(a, b, c, d):
    return Conjunction([b-c, d-a])
def always(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    conjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions.append(Disjunction([noIntersection(t_j, t_j_1, t_i + a, t_i_1 + b), zphis[j]]))
    return Conjunction(conjunctions)
def eventually(i, a, b, zphis, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        disjunctions.append(Conjunction([hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphis[j]]))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])
def until(i, a, b, zphi1s, zphi2s, PWL):
    t_i = PWL[i][1]
    t_i_1 = PWL[i+1][1]
    z_intervalWidth = b-a-(t_i_1-t_i)-EPS
    disjunctions = []
    for j in range(len(PWL)-1):
        t_j = PWL[j][1]
        t_j_1 = PWL[j+1][1]
        conjunctions = [hasIntersection(t_j, t_j_1, t_i_1 + a, t_i + b), zphi2s[j]]
        for l in range(j+1):
            t_l = PWL[l][1]
            t_l_1 = PWL[l+1][1]
            conjunctions.append(Disjunction([noIntersection(t_l, t_l_1, t_i, t_i_1 + b), zphi1s[l]]))
        disjunctions.append(Conjunction(conjunctions))
    return Conjunction([z_intervalWidth, Disjunction(disjunctions)])
def mu(i, PWL, bloat_factor, A, b):
    bloat_factor = np.max([0, bloat_factor])
    b = b.reshape(-1)
    num_edges = len(b)
    conjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        for j in [i, i+1]:
            x = PWL[j][0]
            conjunctions.append(b[e] - np.linalg.norm(a) * bloat_factor - sum([a[k]*x[k] for k in range(len(x))]) - EPS)
    return Conjunction(conjunctions)
def negmu(i, PWL, bloat_factor, A, b):
    b = b.reshape(-1)
    num_edges = len(b)
    disjunctions = []
    for e in range(num_edges):
        a = A[e,:]
        conjunctions = []
        for j in [i, i+1]:
            x = PWL[j][0]
            conjunctions.append(sum([a[k]*x[k] for k in range(len(x))]) - (b[e] + np.linalg.norm(a) * bloat_factor) - EPS)
        disjunctions.append(Conjunction(conjunctions))
    return Disjunction(disjunctions)
def add_space_constraints(model, xlist, limits, bloat=0.):
    xlim, ylim = limits
    for x in xlist:
        model.addConstr(x[0] >= (xlim[0] + bloat))
        model.addConstr(x[1] >= (ylim[0] + bloat))
        model.addConstr(x[0] <= (xlim[1] - bloat))
        model.addConstr(x[1] <= (ylim[1] - bloat))
    return None
def add_time_constraints(model, PWL, tmax=None):
    if tmax is not None:
        model.addConstr(PWL[-1][1] <= tmax - T_MIN_SEP)
    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        model.addConstr(t2 - t1 >= T_MIN_SEP)
def add_velocity_constraints(model, PWL, vmax=3):
    for i in range(len(PWL)-1):
        x1, t1 = PWL[i]
        x2, t2 = PWL[i+1]
        L1_dist = L1Norm(model, _sub(x1,x2))
        model.addConstr(L1_dist <= vmax * (t2 - t1))
class Node(object):
    """docstring for Node"""
    def __init__(self, op, deps = [], zs = [], info = []):
        super(Node, self).__init__()
        self.op = op
        self.deps = deps
        self.zs = zs
        self.info = info
def handleSpecTree(spec, PWL, bloat_factor, size):
    for dep in spec.deps:
        handleSpecTree(dep, PWL, bloat_factor, size)
    if len(spec.zs) == len(PWL)-1:
        return
    elif len(spec.zs) > 0:
        raise ValueError('incomplete zs')
    if spec.op == 'mu':
        spec.zs = [mu(i, PWL, 0.1, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'negmu':
        spec.zs = [negmu(i, PWL, bloat_factor + size, spec.info['A'], spec.info['b']) for i in range(len(PWL)-1)]
    elif spec.op == 'and':
        spec.zs = [Conjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'or':
        spec.zs = [Disjunction([dep.zs[i] for dep in spec.deps]) for i in range(len(PWL)-1)]
    elif spec.op == 'U':
        spec.zs = [until(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, spec.deps[1].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'F':
        spec.zs = [eventually(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    elif spec.op == 'A':
        spec.zs = [always(i, spec.info['int'][0], spec.info['int'][1], spec.deps[0].zs, PWL) for i in range(len(PWL)-1)]
    else:
        raise ValueError('wrong op code')
def gen_CDTree_constraints(model, root):
    if not hasattr(root, 'deps'):
        return [root,]
    else:
        if len(root.constraints)>0:
            return root.constraints
        dep_constraints = []
        for dep in root.deps:
            dep_constraints.append(gen_CDTree_constraints(model, dep))
        zs = []
        for dep_con in dep_constraints:
            if isinstance(root, Disjunction):
                z = model.addVar(vtype=GRB.BINARY)
                zs.append(z)
                dep_con = [con + M * (1 - z) for con in dep_con]
            root.constraints += dep_con
        if len(zs)>0:
            root.constraints.append(sum(zs)-1)
        model.update()
        return root.constraints
def add_CDTree_Constraints(model, root):
    constrs = gen_CDTree_constraints(model, root)
    for con in constrs:
        model.addConstr(con >= 0)

''' API documentation
- INPUTS:
x0s = [[x0, y0]] // initial state
specs = [STL formula] // STL spec: e.g., stl_and, reach, avoid, stl_until, inside, outside
limits = [[x_min, x_max], [y_min, y_max]] // map boundary
- OUTPUTS:
PWLs [[[x0,y0,t0], [x1,y1,t1], ..., [xk,yk,tk]]] // 1x(k+1)x3 timed waypoints

- EXAMPLES:
// remember first to import these functions
from apis.milp.py import stl_and, reach, avoid, stl_until, outside, inside
// assume env.obj_A = [x, y, w, h]
// Init at (0,1), STL="Reach A at time [5, 10] and avoid B at time [0, 20]", map range -8<=x<=8, -6<=x<=6
PWLs=solve_sequence([[0, 1]], [stl_and(reach(env.obj_A, 5, 10), avoid(env.obj_B, 0, 20))], limits=[[-8., 8.],[-6., 6.]])
// assume env.obstacles[0] = [2, x, y, w, h]
// Init at (2,3), STL="Do not reach A, until first reach B from [0, nt]"
PWLs=solve_sequence([[2, 3]], [stl_until(outside(env.obstacles[0][1:]), inside(env.obstacles[1][1:]), 0, nt)])
'''
def solve_sequence(x0s, specs, limits=None):
    num_segs = 12
    vmax = 3.
    PWLs = []
    m = Model("xref")
    m.setParam(GRB.Param.IntFeasTol, IntFeasTol)
    m.setParam(GRB.Param.MIPGap, 1e-4)
    for idx_a in range(len(x0s)):
        x0 = x0s[idx_a]
        x0 = np.array(x0).reshape(-1).tolist()
        spec = specs[idx_a]
        dims = len(x0)
        PWL = []
        for i in range(num_segs+1):
            PWL.append([m.addVars(dims, lb=-GRB.INFINITY), m.addVar()])
        PWLs.append(PWL)
        m.update()
        m.addConstrs(PWL[0][0][i] == x0[i] for i in range(dims))
        m.addConstr(PWL[0][1] == 0)
        if limits is not None:
            add_space_constraints(m, [P[0] for P in PWL], limits)
        add_velocity_constraints(m, PWL, vmax=vmax)
        handleSpecTree(spec, PWL, 0.2, 0.22)
        add_CDTree_Constraints(m, spec.zs[0])
    obj = sum([PWL[-1][1] for PWL in PWLs])
    m.setObjective(obj, GRB.MINIMIZE)
    m.write("test.lp")
    try:
        start = time.time()
        m.optimize()
        end = time.time()
        print('sovling it takes %.3f s'%(end - start))
        PWLs_output = []
        for PWL in PWLs:
            PWL_output = []
            for P in PWL:
                PWL_output.append([P[0][0].X, P[0][1].X, P[1].X])
            PWLs_output.append(PWL_output)
        m.dispose()
        return PWLs_output
    except Exception as e:
        print(e)
        m.dispose()
    return [None,]
