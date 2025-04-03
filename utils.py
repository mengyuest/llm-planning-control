import os
from os.path import join as ospj
import sys
import time
import shutil
from datetime import datetime
import random
import numpy as np
import torch
import matplotlib.pyplot as plt

############################### 
THE_EXP_ROOT_DIR="exps"  # 
###############################

import torch

def plt_save_close(img_path, bbox_inches='tight', pad_inches=0.1):
    plt.savefig(img_path, bbox_inches=bbox_inches, pad_inches=pad_inches)
    plt.close()

def get_exp_dir():
    return THE_EXP_ROOT_DIR

def to_np(x):
    return x.detach().cpu().numpy()

def to_torch(x):
    return torch.from_numpy(x).float().cuda()

def uniform_tensor(amin, amax, size):
    return torch.rand(size) * (amax - amin) + amin

def rand_choice_tensor(choices, size):
    return torch.from_numpy(np.random.choice(choices, size)).float()

def uniform(a, b, size):
    return torch.rand(*size) * (b - a) + a

def linspace(a, b, size):
    return torch.from_numpy(np.linspace(a, b, size)).float()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def write_cmd_to_file(log_dir, argv):
    with open(ospj(log_dir, "cmd.txt"), "w") as f:
        f.write("python " + " ".join(argv))

def time_format(secs):
    _s = secs % 60 
    _m = secs % 3600 // 60
    _h = secs % 86400 // 3600
    _d = secs // 86400
    if _d != 0:
        return "%02dD%02dh%02dm%02ds"%(_d, _h, _m, _s)
    else:
        if _h != 0:
            return "%02dH%02dm%02ds"%(_h, _m, _s)
        else:
            if _m != 0:
                return "%02dm%02ds"%(_m, _s)
            else:
                return "%05.2fs"%(_s)

# TODO create the exp directory
def setup_exp_and_logger(args, set_gpus=True, just_local=False):
    seed_everything(args.seed)
    if set_gpus and hasattr(args, "gpus") and args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    sys.stdout = logger = Logger()
    EXP_ROOT_DIR = get_exp_dir()
    if hasattr(args, "local") and args.local:
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, args.exp_name)
    else:
        args.exp_dir_full = os.path.join(EXP_ROOT_DIR, "g%s_%s" % (logger._timestr, args.exp_name))
    args.viz_dir = os.path.join(args.exp_dir_full, "viz")
    args.src_dir = os.path.join(args.exp_dir_full, "src")
    args.model_dir = os.path.join(args.exp_dir_full, "models")
    os.makedirs(args.viz_dir, exist_ok=True)
    os.makedirs(args.src_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    for fname in os.listdir('./'):
        if fname.endswith('.py'):
            shutil.copy(fname, os.path.join(args.src_dir, fname))

    logger.create_log(args.exp_dir_full)
    write_cmd_to_file(args.exp_dir_full, sys.argv)
    np.savez(os.path.join(args.exp_dir_full, 'args'), args=args)

    return args


# TODO logger
class Logger(object):
    def __init__(self):
        self._terminal = sys.stdout
        self._timestr = datetime.fromtimestamp(time.time()).strftime("%m%d-%H%M%S")

    def create_log(self, log_path):
        self.log = open(log_path + "/log-%s.txt" % self._timestr, "a", 1)

    def write(self, message):
        self._terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class EtaEstimator():
    def __init__(self, start_iter, end_iter, check_freq=1):
        self.start_iter = start_iter
        self.end_iter = end_iter
        self.check_freq = check_freq
        self.curr_iter = start_iter
        self.start_timer = None
        self.interval = 0
        self.eta_t = 0

    def update(self):
        if self.start_timer is None:
            self.start_timer = time.time()
        self.curr_iter += 1
        self.interval = self.elapsed() / (self.curr_iter - self.start_iter)        
        self.eta_t = self.interval * (self.end_iter - self.curr_iter)

    def elapsed(self):
        return time.time() - self.start_timer
    
    def eta(self):
        return self.eta_t
    
    def elapsed_str(self):
        return time_format(self.elapsed())
    
    def interval_str(self):
        return time_format(self.interval)

    def eta_str(self):
        return time_format(self.eta_t)