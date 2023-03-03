# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

import pdb
import json
import argparse
import math
import autograd.numpy as np
import autograd
import torch
from torch import nn
from torchdiffeq import odeint
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
import os, sys
import standard_io as std_io

matplotlib.use('agg')
torch.set_default_dtype(torch.float64)
sns.set()
sns.set_style('whitegrid')
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

xmin = -3.2; xmax = 3.2; ymin = -3.2; ymax = 3.2
DPI = 200
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 100
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2

def get_args():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--test_samples', default=25, type=int)
    parser.add_argument('--T', default=5, type=int, help='observation period')
    parser.add_argument('--radius_a', default=1., type=float)
    parser.add_argument('--radius_b', default=1., type=float)
    parser.add_argument('--timescale', default=15, type=int, help='number of observations per a second')
    parser.add_argument('--name', default='pendulum', type=str)
    parser.add_argument('--gridsize', default=15, type=int)
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--eta', default=0.05, type=float)
    parser.add_argument('--save_dir', default=THIS_DIR, type=str)
    parser.add_argument('--input_dim', default=4, type=int)
    return parser.parse_args()

class ODE_d_pendulum(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.M = self.permutation_tensor(input_dim)

    def forward(self, t, x):
        H = self.H(x)
        dH = torch.autograd.grad(H.sum(), x)[0]
        field = dH @ self.M.t()
        dH[:,0] = 0; dH[:,1] = 0
        field = field - args.eta * dH
        return field
    
    def time_derivative(self, x):
        H = self.H(x)
        dH = torch.autograd.grad(H.sum(), x)[0]
        field = dH @ self.M.t()
        dH[:,0] = 0; dH[:,1] = 0
        field = field - args.eta * dH
        return field
        
    def H(self, coords):
        if len(coords) == 4:
            q1, q2, p1, p2 = coords[0], coords[1], coords[2], coords[3]
        else:
            q1, q2, p1, p2 = coords[:,0], coords[:,1], coords[:,2], coords[:,3]
        m1 = .2; m2 = .1
        H = ( (m2*p1**2 + (m1+m2)*p2**2 - 2*m2*p1*p2*torch.cos(q1-q2))
              / (2*m2*(m1+m2*torch.sin(q1-q2)**2))
              - (m1+m2)*9.8*torch.cos(q1)
              - m2*9.8*torch.cos(q2) )
        return H

    def permutation_tensor(self,n):
        M = torch.eye(n)
        M = torch.cat([M[n//2:], -M[:n//2]])
        return M

def vis_obs(save_dir, data, trajectory_name):
    y = data[trajectory_name]
    t_eval = data['t']
    t = torch.tensor(np.linspace(0, t_eval[-1], t_eval.shape[0]))
    
    fig = plt.figure(figsize=(21,11.3), facecolor='white', dpi=DPI)
    N = y.shape[0]
    N = 28 if N > 28 else N
    for i in range(N):
        xs1 = y[i,:,0]; xs2 = y[i,:,1]

        ax = fig.add_subplot(math.ceil(N/7), 7, i+1, frameon=True)
        ax.scatter(t_eval, xs1, s=0.5)
        ax.scatter(t_eval, xs2, s=0.5)
        plt.axis([t_eval.min().item(), t_eval.max().item(), -3, 3])
        plt.xlabel("$t$", fontsize=12)
        plt.ylabel("$x_q$", rotation=0, fontsize=12)
        plt.title("Sample " + str(i+1))
        plt.grid(False)
    plt.tight_layout()
    fig.savefig('{}/{}.{}'.format(save_dir, trajectory_name, FORMAT))
    plt.close()

def vis_energy(save_dir, data, e_name):
    es = data[e_name]# * -1
    t_eval = data['t']
    
    fig = plt.figure(figsize=(11.3, 21), facecolor='white', dpi=DPI)
    N = es.shape[0]
    N = 28 if N > 28 else N
    for i in range(N):
        t = t_eval
        ax = fig.add_subplot(math.ceil(N/4), 4, i+1, frameon=True)
        ax.plot(t, es[i],'-', color='black')
        ymax = es.min()

        ax.axis([0, args.T, 0, ymax*1.5])
        plt.xlabel("time", fontsize=12)
        plt.ylabel("Energy", rotation=90, fontsize=12)
        plt.title("Sample " + str(i+1))
    plt.tight_layout()
    fig.savefig('{}/{}.{}'.format(save_dir, e_name, FORMAT))
    plt.close()

def get_field(ode, xmin, xmax, ymin, ymax, gridsize):
    field = {}

    # meshgrid to get vector field
    b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
    x = np.stack([b.flatten(), a.flatten()]).T
    x = torch.tensor(x, requires_grad=True)

    # get vector directions
    dx = ode.time_derivative(x)
    field['x'] = x.detach().numpy()
    field['dx'] = dx.detach().numpy()
    field['mesh_a'] = a
    field['mesh_b'] = b
    return field

def vis_field(save_dir, field, name):
    a = field['mesh_a']
    b = field['mesh_b']
    fig = plt.figure(figsize=(4,3), facecolor='white', dpi=DPI)
    ax = fig.subplots()
    ax.set_aspect('equal', adjustable='box')
    scale = ARROW_SCALE
    ax.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
              scale=scale, width=ARROW_WIDTH)
    plt.tight_layout()
    fig.savefig('{}/{}.{}'.format(save_dir, name, FORMAT))
    plt.close()

def get_init():
    N = args.test_samples
    if args.name in ['d_pendulum']:
        x0s = 2*np.random.uniform(0, 1, N*2) - 1
        x0s = np.hstack([x0s.reshape(N,2), np.zeros([N,2])])
        x0s = torch.tensor(x0s, requires_grad=True)
    return x0s

def path_arrange(path):
  x = []
  for i in range(path.shape[1]):
    x.append(path[:,i,:])
  return torch.stack(x)

def generate_data(ode):

    data = {}
    np.random.seed(args.seed)
    
    xs, es = [], []
    x0s = get_init()
    t = torch.tensor(np.linspace(0, args.T, int(args.timescale*args.T+1)))
    xs = odeint(ode, x0s, t, method='dopri5', atol=1e-8, rtol=1e-8)
    xs = path_arrange(xs)
    for x in xs:
        e = ode.H(x)
        es.append(e)
    es = torch.stack(es)

    data['xs'] = xs.detach().numpy()
    data['t'] = t.detach().numpy()
    data['es'] = es.detach().numpy()

    vis_obs(save_dir, data, trajectory_name='xs')
    vis_energy(save_dir, data, e_name='es')
    
    return data


if __name__ == "__main__":

    args = get_args()
    input_dim = args.input_dim
    if args.name == 'd_pendulum':
        ode = ODE_d_pendulum(input_dim)

    save_dir = args.save_dir + '/' + args.name + '/' + str(args.eta) + '/test'
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    data = generate_data(ode)
        
    std_io.pkl_write(save_dir + '/data.pkl', data)

    filename = '{}/{}.json'.format(save_dir, args.name)
    with open(filename, 'w') as f:
        json.dump(vars(args), f)
