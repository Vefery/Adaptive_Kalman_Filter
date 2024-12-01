import torch
import matplotlib.pyplot as plt
from torch.distributions.normal import Normal
from math import sqrt

class KF():
    def __init__(self, A :torch.Tensor, H :torch.Tensor, Q :torch.Tensor, R :torch.Tensor, x0 :torch.Tensor):
        self.xk = x0.float().reshape(-1, 1)
        self.A = A.float()
        if self.A.dim() == 1:
            self.A = self.A.unsqueeze(0)

        self.H = H.float()
        if self.H.dim() == 1:
            self.H = self.H.unsqueeze(0)

        self.Q = Q
        self.R = R
        self.Pk = (x0 - self.xk) @ (x0 - self.xk).T
        self.A_tr = self.A.T
        self.H_tr = self.H.T
        self.identity = torch.eye(x0.shape[0], dtype=torch.float32, requires_grad=False)

        self.n = self.xk.shape[0]
        self.m = self.H.shape[0]
        
    def predict(self):
        xk_new_noise = self.A @ self.xk
        Pk_new_noise = self.A @ self.Pk @ self.A_tr + self.Q

        return xk_new_noise, Pk_new_noise

    def correct(self, xk_new_noise :torch.Tensor, Pk_new_noise :torch.Tensor, zk :torch.Tensor):
        Kk = Pk_new_noise @ self.H_tr @ (self.H @ Pk_new_noise @ self.H_tr + self.R).inverse()
        xk_new = xk_new_noise + Kk @ (zk - self.H @ xk_new_noise)
        Pk_new = (self.identity - Kk @ self.H) @ Pk_new_noise

        self.xk = xk_new
        self.Pk = Pk_new

        return xk_new

class AEKF():
    def __init__(self, A :torch.Tensor, H :torch.Tensor, Q0 :torch.Tensor, R0 :torch.Tensor, x0 :torch.Tensor, z0 :torch.Tensor, alpha = 0.3):
        self.xk = x0.float().reshape(-1, 1)
        self.A = A.float()
        if self.A.dim() == 1:
            self.A = self.A.unsqueeze(0)

        self.H = H.float()
        if self.H.dim() == 1:
            self.H = self.H.unsqueeze(0)

        self.Qk = Q0.float()
        self.Rk = R0.float()
        self.Pk = (x0 - self.xk) @ (x0 - self.xk).T
        self.A_tr = self.A.T
        self.H_tr = self.H.T
        self.identity = torch.eye(x0.shape[0], dtype=torch.float32, requires_grad=False)
        self.alpha = alpha

        self.n = self.xk.shape[0]
        self.m = self.H.shape[0]

        t = z0.view(-1, 1) - self.H @ self.xk
        self.eps_sum = t @ t.T
        
    def predict(self):
        xk_new_noise = self.A @ self.xk
        Pk_new_noise = self.A @ self.Pk @ self.A_tr + self.Qk

        return xk_new_noise, Pk_new_noise

    def correct(self, xk_new_noise :torch.Tensor, Pk_new_noise :torch.Tensor, zk :torch.Tensor, k :int):
        dk = zk - self.H @ xk_new_noise
        eps_est = self.eps_sum / k
        Rk_new = self.alpha * self.Rk + (1.0 - self.alpha) * (eps_est + self.H @ Pk_new_noise @ self.H_tr)
        Kk = Pk_new_noise @ self.H_tr @ (self.H @ Pk_new_noise @ self.H_tr + Rk_new).inverse()
        xk_new = xk_new_noise + Kk @ dk
        Pk_new = (self.identity - Kk @ self.H) @ Pk_new_noise

        Qk_new = self.alpha * self.Qk + (1.0 - self.alpha) * (Kk @ dk @ dk.T @ Kk.T)

        eps = zk - self.H @ xk_new
        self.eps_sum += eps @ eps.T

        self.xk = xk_new
        self.Pk = Pk_new
        self.Qk = Qk_new
        self.Rk = Rk_new

        return xk_new, Qk_new, Rk_new

delta_t = 1.0
total = 1000

A = torch.tensor([[1, 0, delta_t, 0], [0, 1, 0, delta_t], [0, 0, 1, 0], [0, 0, 0, 1]])
H = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=torch.float32)
Q = 0.01 * torch.diag(torch.tensor([1, 1, 1, 1]))
R = 0.1 * torch.diag(torch.tensor([1.0, 1.0]))
x0 = torch.tensor([[1], [1], [0], [0]], dtype=torch.float32)

sigma_v = sqrt(0.1)
dist = Normal(0, sigma_v)
z0 = H @ x0 + dist.sample((R.dim(), 1))

aekf = AEKF(A, H, Q, R, x0, z0, alpha=0.3)
kf = KF(A, H, Q, R, x0)

xerr = []
kf_xerr = []
x_prev = x0
for k in range(1, total):
    x_cur = A @ x_prev

    xk_new_noise, pk_new_noise = aekf.predict()
    kf_xk_new_noise, kf_pk_new_noise = kf.predict()

    zk = H @ x_cur

    xk, Qk, Rk = aekf.correct(xk_new_noise, pk_new_noise, zk + dist.sample((aekf.m, 1)), k)
    xk_fk = kf.correct(kf_xk_new_noise, kf_pk_new_noise, zk + dist.sample((aekf.m, 1)))

    zk_aekf = H @ xk
    zk_kf = H @ xk_fk
    if k % 10 == 0:
        xerr.append(torch.linalg.norm((zk - zk_aekf)**2 / zk.numel()))
        kf_xerr.append(torch.linalg.norm((zk - zk_kf)**2 / zk.numel()))

x_values = list(range(len(xerr)))
plt.figure(figsize=(10, 5))
plt.plot(x_values, xerr, marker='o', color='blue')
plt.plot(x_values, kf_xerr, marker='.', color='green')

plt.title('Error')
plt.xlabel('Time step')
plt.ylabel('Error')

plt.grid()
plt.show()