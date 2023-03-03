# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

import math
import numpy as np
import os, torch, pickle, zipfile
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
import standard_io as std_io
torch.set_default_dtype(torch.float64)

DPI = 200
FORMAT = 'pdf'
LINE_SEGMENTS = 10
ARROW_SCALE = 100
ARROW_WIDTH = 6e-3
LINE_WIDTH = 2
xmin = -3.2; xmax = 3.2; ymin = -3.2; ymax = 3.2


def get_batch(args, x, t_eval, batch_step):
  n_samples, n_points, input_dim = x.shape
  N = n_samples
  n_ids = torch.from_numpy(np.arange(N))
  p_ids = torch.from_numpy(np.random.choice(np.arange(n_points-batch_step, dtype=np.int64), N, replace=True))
  batch_x0 = x[n_ids,p_ids].reshape([N,1,input_dim])
  batch_step += 1
  batch_t = t_eval[:batch_step]
  batch_x = ( torch.stack([x[n_ids, p_ids+i] for i in range(batch_step)], dim=0)
              .reshape([batch_step,N,1,input_dim]) )
  return batch_x0, batch_t, batch_x

def arrange(args, x, t_eval):
  n_samples, n_points, input_dim = x.shape
  n_ids = np.arange(n_samples, dtype=np.int64)
  p_ids = np.array([0]*n_samples)
  batch_x0 = x[n_ids,p_ids].reshape([n_samples,1,input_dim])
  batch_t = t_eval
  batch_x = torch.stack([x[n_ids, p_ids+i] for i in range(n_points)],dim=0).reshape([n_points,n_samples,1,input_dim])
  return batch_x0, batch_t, batch_x

def get_field(func, xmin, xmax, ymin, ymax, gridsize):
  field = {'meta': locals()}

  # meshgrid to get vector field
  b, a = np.meshgrid(np.linspace(xmin, xmax, gridsize), np.linspace(ymin, ymax, gridsize))
  ys = np.stack([b.flatten(), a.flatten()])
  ys = torch.tensor( ys, dtype=torch.float64, requires_grad=True).t()

  # get vector directions
  dydt = func(torch.tensor([0]),ys)
  field['x'] = ys.cpu().detach().numpy()
  field['dx'] = dydt.squeeze().cpu().detach().numpy()
  field['mesh_a'] = a
  field['mesh_b'] = b
  return field

def vis_path(filename, field, y, t, xmin, xmax, ymin, ymax):
  fig = plt.figure(figsize=(21, 11.3), facecolor='white', dpi=DPI)
  N = y.shape[0]
  N = 28 if N > 28 else N
  for i in range(N):
    ax = fig.add_subplot(math.ceil(N/7), 7, i+1, frameon=True)
    ax.set_aspect('equal', adjustable='box')
    ax.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
              scale=ARROW_SCALE, width=ARROW_WIDTH,
              cmap='gray_r', color=(.5,.5,.5))
    ax.scatter(y[i][:,0], y[i][:,1], c=t, s=0.5, cmap='coolwarm')
    plt.axis([xmin, xmax, ymin, ymax])
    plt.xlabel("$x_q$", fontsize=12)
    plt.ylabel("$x_p$", rotation=0, fontsize=12)
    plt.title("Sample " + str(i+1))
    plt.grid(False)
  plt.tight_layout()
  fig.savefig(filename)
  plt.close()

def vis_path_2d(filename, y, t_eval):
  fig = plt.figure(figsize=(21, 11.3), facecolor='white', dpi=DPI)
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
  fig.savefig(filename)
  plt.close()

def vis_field(filename, field, xmin, xmax, ymin, ymax):
  fig = plt.figure(figsize=(4,3), facecolor='white', dpi=DPI)
  ax = fig.subplots()
  ax.set_aspect('equal', adjustable='box')
  scale = ARROW_SCALE
  ax.quiver(field['x'][:,0], field['x'][:,1], field['dx'][:,0], field['dx'][:,1],
            scale=scale, width=ARROW_WIDTH,
            cmap='gray_r', color=(.5,.5,.5))
  plt.axis([xmin, xmax, ymin, ymax])
  plt.xlabel("$x_q$", fontsize=12)
  plt.ylabel("$x_p$", rotation=0, fontsize=12)
  plt.grid(False)
  plt.tight_layout()
  fig.savefig(filename)
  plt.close()

def vis_err(filename, es, t):
  fig = plt.figure(figsize=(21, 11.3), facecolor='white', dpi=DPI)
  N = es.shape[0]
  N = 28 if N > 28 else N
  for i in range(N):
    ax = fig.add_subplot(math.ceil(N/7), 7, i+1, frameon=True)
    ax.plot(t, es[i],'-', color='black')
    ax.axis([0, t.max(), 0, es.max()*1.2])
    plt.xlabel("time", fontsize=12)
    plt.ylabel("MSE", rotation=90, fontsize=12)
    plt.title("Sample " + str(i+1))
  plt.tight_layout()
  fig.savefig(filename)
  plt.close()

def vis_energy(filename, true, es, t):
  fig = plt.figure(figsize=(21, 11.3), facecolor='white', dpi=DPI)
  N = es.shape[0]
  N = 28 if N > 28 else N
  if es.max() > 0:
    ymax = es.max() if true.max() < es.max() else true.max()
  else:
    ymax = es.min() if true.min() > es.min() else true.min()
  for i in range(N):
    ax = fig.add_subplot(math.ceil(N/7), 7, i+1, frameon=True)
    ax.plot(t, true[i],'-', color='black')
    ax.plot(t, es[i],'-', color='red')
    ax.axis([0, t.max(), 0, ymax*1.2])
    plt.xlabel("time", fontsize=12)
    plt.ylabel("Energy", rotation=90, fontsize=12)
    plt.title("Sample " + str(i+1))
  plt.tight_layout()
  fig.savefig(filename)
  plt.close()

def path_arrange(path):
  x = []
  for i in range(path.shape[1]):
    x.append(path[:,i,:])
  return torch.stack(x)

def d_pendulum_energy(coords):
  q1, q2, p1, p2 = coords[:,:,0], coords[:,:,1], coords[:,:,2], coords[:,:,3]
  m1 = .2; m2 = .1
  H = ( (m2*p1**2 + (m1+m2)*p2**2 - 2*m2*p1*p2*np.cos(q1-q2))
        / (2*m2*(m1+m2*np.sin(q1-q2)**2))
        - (m1+m2)*9.8*np.cos(q1)
        - m2*9.8*np.cos(q2) )
  return H

def pendulum_energy(coords):
    qs = coords[:,:,0]; ps = coords[:,:,1]
    energy = 3*(1-np.cos(qs)) + ps**2
    return energy

def duffing_energy(coords):
    qs = coords[:,:,0]; ps = coords[:,:,1]
    energy= .5*ps**2 - .5*qs**2 + .25*qs**4
    return energy

def real_pend_energy(coords):
    qs = coords[:,:,0]; ps = coords[:,:,1]
    energy = 2.4*(1-np.cos(qs)) + ps**2
    return energy
