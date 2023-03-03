# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#num_threads = '1'
#os.environ['OMP_NUM_THREADS'] = num_threads
#os.environ['MKL_NUM_THREADS'] = num_threads
#os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import os, sys
import copy
import pdb
import time
import json
import argparse
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch.autograd import detect_anomaly
from torchdiffeq import odeint
import standard_io as std_io
import standard_vis as std_vis

from models import SSGP
from utils import *

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
torch.set_default_dtype(torch.float64)
DPI = 200
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 100
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
gridsize = 15
xmin = -3.2; xmax = 3.2; ymin = -3.2; ymax = 3.2

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--name', default='pendulum', type=str)
    parser.add_argument('--s', default=0, type=int, help='dataset index')
    parser.add_argument('--gridsize', default=15, type=int, help='gridsize')
    parser.add_argument('--sigma', default=0.1, type=float, help='noise variance')
    parser.add_argument('--eta', default=0.1, type=float)
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('--timescale', default=2, type=int)
    parser.add_argument('--num_basis', default=200, type=int)
    parser.add_argument('--friction', action='store_true')
    parser.add_argument('--task2', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":

    # setting
    args = get_args()
    label = 'SSGP/' + str(args.num_basis)
    i_dir = ( '../data/' + args.name + '/' + str(args.eta) + '/result/' + str(args.sigma)
              + '/' + str(args.samples) + '/' + str(args.timescale) + '/' + str(args.s) + '/' + label)
    save_dir = ( '../data/' + args.name + '/' + str(args.eta) + '/result/' + str(args.sigma) 
                 + '/' + str(args.samples) + '/' + str(args.timescale) + '/' + str(args.s)
                 + '/' + label)
    save_dir += '/test_task2' if args.task2 else '/test_task1'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    eta = 0.0 if args.task2 else args.eta
    filename = '../data/' + args.name + '/' + str(eta) + '/test/data.pkl'
    data = std_io.pkl_read(filename)

    # test data
    xs = torch.tensor(data['xs'])
    x0s = xs[:,0,:]
    t = torch.tensor(data['t'])
    es = data['es']
    n_samples, n_points, input_dim = xs.shape
    output_dim = input_dim

    # learned model
    model_json = open(i_dir + '/model.json')
    hyperparam = json.load(model_json)
    num_basis = hyperparam['num_basis']
    model = SSGP(input_dim, num_basis, args.friction).double()
    model.load_state_dict(torch.load(i_dir + '/model.tar'), False)
    if args.task2:
        model.state_dict()['eta'][0] = torch.tensor([0.0]) 

    # simulation
    model.mean_w()
    if input_dim == 2:
        pred_field = get_field(model.forward, xmin, xmax, ymin, ymax, gridsize)
    x0s = x0s.reshape([x0s.shape[0],1,x0s.shape[1]])
    pred = odeint(model, x0s, t, method='dopri5', atol=1e-8, rtol=1e-8)

    # output(trajectory)
    pred = path_arrange(pred.squeeze())
    pred = pred.detach().numpy()
    filename = save_dir + '/pred.npy'
    np.save(filename, pred)
    filename = save_dir + '/pred.pdf'
    if input_dim == 2:
        vis_path(filename, pred_field, pred, t, xmin, xmax, ymin, ymax)
    else:
        vis_path_2d(filename, pred, t)

    # output(trajectory MSE)
    true = xs.detach().numpy()
    traj_e = ((true-pred)**2).sum(axis=2)
    filename = save_dir + '/traj_SE.npy'
    np.save(filename, traj_e)
    filename = save_dir + '/traj_SE.pdf'
    vis_err(filename, traj_e, t)
    traj_E = traj_e.mean()
    filename = save_dir + '/traj_MSE.csv'
    std_io.csv_write(filename, np.array([traj_E]))

    # output(energy)
    true = es
    if args.name == 'pendulum':
        e_pred = pendulum_energy(pred)
    elif args.name == 'duffing':
        e_pred = duffing_energy(pred)
    elif args.name == 'd_pendulum':
        e_pred = d_pendulum_energy(pred)
    filename = save_dir + '/e_pred.npy'
    np.save(filename, e_pred)
    filename = save_dir + '/e_pred.pdf'
    vis_energy(filename, true, e_pred, t)

    # output(energy MSE)
    ener_e = (true-e_pred)**2
    filename = save_dir + '/ener_SE.npy'
    np.save(filename, ener_e)
    filename = save_dir + '/ener_SE.pdf'
    vis_err(filename, ener_e, t)
    ener_E = ener_e.mean()
    filename = save_dir + '/ener_MSE.csv'
    std_io.csv_write(filename, np.array([ener_E]))
