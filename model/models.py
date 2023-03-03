# Symplectic Spectrum Gaussian Processes | 2022
# Yusuke Tanaka

import sys
import pdb
import torch
import torch.nn as nn
import numpy as np
import math
from sqrtm import sqrtm
torch.set_default_dtype(torch.float64)


class SSGP(torch.nn.Module):
  def __init__(self, input_dim, basis, friction):
    super(SSGP, self).__init__()
    self.sigma = nn.Parameter(torch.tensor([1e-1]))
    self.a = nn.Parameter(torch.ones(input_dim)*1e-1)
    self.b = nn.Parameter(1e-4 * (torch.rand(basis*2)-0.5))
    self.init_C(basis)
    self.sigma_0 = nn.Parameter(torch.tensor([1e-0]))
    self.lam = nn.Parameter(torch.ones(input_dim)*1.5)
    if friction:
      self.eta = nn.Parameter(torch.tensor([1e-16]))
    else:
      self.eta = torch.tensor([0.0])
    self.M = self.permutation_tensor(input_dim)
    np.random.seed(0)
    tmp = torch.tensor(np.random.normal(0, 1, size=(int(basis/2.), input_dim)))
    self.epsilon = torch.vstack([tmp,-tmp])
    self.d = input_dim
    self.num_basis = basis

  def sampling_epsilon_f(self):
    C = self.make_C()
    sqrt_C = sqrtm(C)
    sqrt_C = torch.block_diag(sqrt_C,sqrt_C)
    epsilon = torch.tensor(np.random.normal(0, 1, size=(1,sqrt_C.shape[0]))).T
    self.w = self.b + (sqrt_C @ epsilon).squeeze()
    num = 99
    for i in range(num):
      epsilon = torch.tensor(np.random.normal(0, 1, size=(1,sqrt_C.shape[0]))).T
      self.w += self.b + (sqrt_C @ epsilon).squeeze()
    self.w = self.w/(num+1)

  def mean_w(self):
    self.w = self.b * 1
    
  def forward(self, t, x):
    s = self.epsilon @ torch.diag((1 / torch.sqrt(4*math.pi**2 * self.lam**2)))
    R = torch.eye(self.d)
    R[:int(self.d/2),:int(self.d/2)] = 0
    mat = 2*math.pi*((self.M-self.eta**2*R)@s.T).T
    x = x.squeeze()
    samples = x.shape[0]
    sim = 2*math.pi*s@x.squeeze().T
    basis_s = -torch.sin(sim); basis_c = torch.cos(sim)

    # deterministic
    tmp = []
    for i in range(self.d):
      tmp.extend([mat[:,i]]*samples)
    tmp = torch.stack(tmp).T
    aug_mat = torch.vstack([tmp,tmp])
    aug_s = torch.hstack([basis_s]*self.d); aug_c = torch.hstack([basis_c]*self.d)
    aug_basis = torch.vstack([aug_s, aug_c])
    PHI = aug_mat * aug_basis
    aug_W = torch.stack([self.w]*samples*self.d).T
    F = PHI * aug_W
    f = torch.vstack(torch.split(F.sum(axis=0),samples)).T
    return f.reshape([samples,1,self.d])

  def neg_loglike(self, batch_x, pred_x):
    n_samples, n_points, dammy, input_dim = batch_x.shape
    likelihood = ( (-(pred_x-batch_x)**2/self.sigma**2/2).nansum()
                   - torch.log(self.sigma**2)/2*n_samples*n_points*input_dim)
    return -likelihood

  def KL_x0(self, x0):
    n, d = x0.shape
    S = torch.diag(self.a**2)
    return .5*((x0*x0).sum() + n*torch.trace(S) - n*torch.logdet(S))

  def KL_w(self):
    num = self.b.shape[0]
    C = self.make_C()
    C = torch.block_diag(C,C)
    term3 = (self.b*self.b).sum() / (self.sigma_0**2 / num * 2)
    term2 = torch.diag(C).sum() / (self.sigma_0**2 / num * 2)
    term1_1 = torch.log(self.sigma_0**2 / num * 2) * num
    term1_2 = torch.logdet(C)
    return .5*( term1_1 - term1_2 + term2 + term3)

  def sampling_x0(self, x0):
    n, dammy, d = x0.shape
    return (x0 + torch.sqrt(torch.stack([self.a**2]*n).reshape([n,1,d]))
            * (torch.normal(0,1, size=(x0.shape[0],1,x0.shape[2]))))

  def permutation_tensor(self,n):
    M = torch.eye(n)
    M = torch.cat([M[n//2:], -M[:n//2]])
    return M

  def init_C(self, basis):
    C = torch.linalg.cholesky(torch.ones(basis,basis)*1e-2+torch.eye(basis)*1e-2)
    C_line = C.reshape([(basis)**2])
    ids = torch.where(C_line!=0)[0]
    self.c = nn.Parameter(C_line[ids])
    ids = []
    for i in range(basis):
      for j in range(i+1):
        ids.append([i,j])
    ids = torch.tensor(ids)
    self.ids0 = ids[:,0]
    self.ids1 = ids[:,1]
    
  def make_C(self):
    C = torch.zeros(self.num_basis,self.num_basis)
    C[self.ids0,self.ids1] = self.c
    C = C@C.T
    return C


