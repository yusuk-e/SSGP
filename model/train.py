# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

#os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
#num_threads = '1'
#os.environ['OMP_NUM_THREADS'] = num_threads
#os.environ['MKL_NUM_THREADS'] = num_threads
#os.environ['NUMEXPR_NUM_THREADS'] = num_threads

import pdb
import os, sys
import copy
import time
import json
import argparse
import math
import numpy as np
import torch
from torchdiffeq import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
xmin = -3.2; xmax = 3.2; ymin = -3.2; ymax = 3.2

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--batch_time', type=int, default=1)
    parser.add_argument('--learn_rate', default=1e-3)
    parser.add_argument('--total_steps', default=2000, type=int)
    parser.add_argument('--print_every', default=100, type=int, help='number of iterations for prints')
    parser.add_argument('--sigma', default=0.1, type=float, help='noise variance')
    parser.add_argument('--eta', default=0.0, type=float)
    parser.add_argument('--samples', default=20, type=int)
    parser.add_argument('--timescale', default=5, type=int)
    parser.add_argument('--name', default='pendulum')
    parser.add_argument('--s', default=0, type=int, help='dataset index')
    parser.add_argument('--gridsize', default=15, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num_basis', default=100, type=int)
    parser.add_argument('--friction', action='store_true')
    return parser.parse_args()

def train():
    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # init model and optimizer
    output_dim = input_dim
    model = SSGP(input_dim, args.num_basis, args.friction).double()
    optim = torch.optim.Adam(model.parameters(), args.learn_rate)

    # train loop
    stats = {'train_loss': [], 'val_loss': []}
    t0 = time.time()
    min_val_loss = 1e+10
    for step in range(args.total_steps+1):

        # train step
        batch_y0, batch_t, batch_ys = get_batch(args, ys, t_eval, batch_step)
        s_batch_x0 = model.sampling_x0(batch_y0)
        model.sampling_epsilon_f()
        pred_x = odeint(model, s_batch_x0, batch_t, method='dopri5', atol=1e-8, rtol=1e-8)
        neg_loglike = model.neg_loglike(batch_ys, pred_x)
        KL_x0 = model.KL_x0(batch_y0.squeeze())
        KL_w = model.KL_w()
        loss = neg_loglike + KL_x0 + KL_w
        loss.backward(); optim.step(); optim.zero_grad()
        train_loss = loss.detach().item()/batch_y0.shape[0]/batch_t.shape[0]
        # run validation data
        with torch.no_grad():
            batch_y0, batch_t, batch_ys = arrange(args, val_ys, t_eval)
            s_batch_x0 = model.sampling_x0(batch_y0)
            model.mean_w()
            pred_val_x = odeint(model, s_batch_x0, t_eval, method='dopri5', atol=1e-8, rtol=1e-8)
            val_neg_loglike = model.neg_loglike(batch_ys, pred_val_x)
            loss = val_neg_loglike
            val_loss = loss.item()/batch_y0.shape[0]/t_eval.shape[0]

        # logging
        stats['train_loss'].append(train_loss)
        stats['val_loss'].append(val_loss)
        if step % args.print_every == 0:
            print("step {}, time {:.2e}, train_loss {:.4e}, val_loss {:.4e}"
                  .format(step, time.time()-t0, train_loss, val_loss))
            t0 = time.time()

        if val_loss < min_val_loss:
            best_model = copy.deepcopy(model)
            min_val_loss = val_loss; best_train_loss = train_loss
            best_step = step
            
    return best_model, stats, best_train_loss, min_val_loss, best_step

def param_save(model):
    std_io.csv_write(save_dir + '/' + 'sigma.csv', model['sigma'].cpu().detach().numpy())
    std_io.csv_write(save_dir + '/' + 'a.csv', model['a'].cpu().detach().numpy())
    std_io.csv_write(save_dir + '/' + 'b.csv', model['b'].cpu().detach().numpy())
    std_io.csv_write(save_dir + '/' + 'c.csv', model['c'].cpu().detach().numpy())
    std_io.csv_write(save_dir + '/' + 'sigma_0.csv', model['sigma_0'].cpu().detach().numpy())
    std_io.csv_write(save_dir + '/' + 'lam.csv', model['lam'].cpu().detach().numpy())
    if args.friction:
        std_io.csv_write(save_dir + '/' + 'eta.csv', model['eta'].cpu().detach().numpy())

    
if __name__ == "__main__":

    # setting
    args = get_args()
    label = 'SSGP/' + str(args.num_basis)
    i_dir = ( '../data/' + args.name + '/' + str(args.eta) + '/train/' + str(args.sigma) 
              + '/' + str(args.samples) + '/' + str(args.timescale) + '/' + str(args.s))
    save_dir = ( '../data/' + args.name + '/' + str(args.eta) + '/result/' + str(args.sigma) 
                 + '/' + str(args.samples) + '/' + str(args.timescale) + '/' + str(args.s) + '/' + label)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None

    # input
    filename = i_dir + '/dataset.pkl'
    data = std_io.pkl_read(filename)
    ys = torch.tensor( data['ys'], requires_grad=False)
    val_ys = torch.tensor( data['val_ys'], requires_grad=False)
    t_eval = torch.tensor( data['t'])
    n_samples, n_points, input_dim = ys.shape
    batch_step = int(((len(t_eval)-1)/t_eval[-1]).item() * args.batch_time)

    # learning
    t0 = time.time()
    model, stats, train_loss, val_loss, step = train()
    train_time = time.time() - t0

    # save
    path = '{}/model.tar'.format(save_dir)
    torch.save(model.state_dict(), path)
    param_save(model.state_dict())
    
    path = '{}/model.json'.format(save_dir)
    with open(path, 'w') as f:
        json.dump(vars(args), f)

    path = '{}/result.csv'.format(save_dir)
    std_io.csv_write(path, np.array(['val_step',step,'train_loss',train_loss,
                                     'val_loss',val_loss,'train_time',train_time]))

    # vis
    ## learning curve
    filename = save_dir + '/learning_curve.pdf'
    x = np.arange(len(stats['train_loss']))
    std_vis.plot(filename, x, [stats['train_loss'],stats['val_loss']],
                  'epoch','neg_loglike', ['train','validation'])

    ## pred field
    if input_dim == 2:
        model.mean_w()
        pred_field = get_field(model.forward, xmin, xmax, ymin, ymax, args.gridsize)
        filename = save_dir + '/pred_field.pdf'
        vis_field(filename, pred_field, xmin, xmax, ymin, ymax)
