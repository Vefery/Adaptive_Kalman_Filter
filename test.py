import torch
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from math import sqrt

class KF():
    def __init__(self, A :torch.Tensor, H :torch.Tensor, Q :torch.Tensor, R :torch.Tensor, x0 :torch.Tensor):
        self.xk = x0.double().detach().clone().reshape(-1, 1)
        self.A = A.double().detach().clone()
        if self.A.dim() == 1:
            self.A = self.A.unsqueeze(0)

        self.H = H.double().detach().clone()
        if self.H.dim() == 1:
            self.H = self.H.unsqueeze(0)

        self.Q = Q.detach().clone()
        self.R = R.detach().clone()
        self.Pk = torch.diag(0.1 * torch.tensor([delta_t, delta_t, delta_t], dtype=torch.float64))
        self.A_tr = self.A.T
        self.H_tr = self.H.T
        self.identity = torch.eye(x0.shape[0], dtype=torch.float64, requires_grad=False)

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
        self.xk = x0.double().detach().clone().reshape(-1, 1)
        self.A = A.double().detach().clone()
        if self.A.dim() == 1:
            self.A = self.A.unsqueeze(0)

        self.H = H.double().detach().clone()
        if self.H.dim() == 1:
            self.H = self.H.unsqueeze(0)

        self.Qk = Q0.double().detach().clone()
        self.Rk = R0.double().detach().clone()

        self.n = self.xk.shape[0]
        self.m = self.H.shape[0]

        self.Pk = torch.eye((self.n), dtype=torch.float64)
        self.A_tr = self.A.T
        self.H_tr = self.H.T
        self.identity = torch.eye(self.n, dtype=torch.float64, requires_grad=False)
        self.alpha = alpha

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
total = 100

# A = torch.tensor([[1, delta_t], [0, 1]], dtype=torch.float64)
# H = torch.tensor([[1, 0]], dtype=torch.float64)
# 0.01 * torch.tensor([[(delta_t ** 3) / 3, (delta_t ** 2) / 2], [(delta_t ** 2) / 2, delta_t]], dtype=torch.float64)
# R = 0.1 * torch.tensor([1.0], dtype=torch.float64)
# x0 = torch.tensor([[0], [0]], dtype=torch.float64)
A = torch.tensor([[1, delta_t, (delta_t ** 2) / 2.0], [0, 1, delta_t], [0, 0, 1]], dtype=torch.float64)
H = torch.tensor([[0, 1, 0]], dtype=torch.float64)
Q = torch.diag(torch.tensor([delta_t, delta_t, delta_t], dtype=torch.float64))
R = torch.tensor([1.0], dtype=torch.float64)
x0 = torch.tensor([[0], [0], [1]], dtype=torch.float64)

measurement_noise = Normal(0.0, R.item())
process_noise = MultivariateNormal(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64), Q)
#process_noise = Normal(0.0, sqrt(0.01 * (delta_t ** 2) / 2))
z0 = H @ x0 + measurement_noise.sample()

aekf = AEKF(A, H, Q, R, x0, z0, alpha=0.3)
kf = KF(A, H, Q, R, x0)

y_real = []
y_noise = []
y_kf = []
y_aekf = []

x_prev = x0
for k in range(1, total):
    x_real = A @ x_prev
    x_prev = x_real.detach().clone()
    x_real += process_noise.sample().view(-1, 1)

    xk_new_noise, pk_new_noise = aekf.predict()
    kf_xk_new_noise, kf_pk_new_noise = kf.predict()

    yk_clean = H @ x_prev
    yk = H @ x_real
    y_real.append(yk_clean)
    yk_noise = yk + measurement_noise.sample()
    y_noise.append(yk_noise)

    xk, Qk, Rk = aekf.correct(xk_new_noise, pk_new_noise, yk_noise, k)
    xk_fk = kf.correct(kf_xk_new_noise, kf_pk_new_noise, yk_noise)

    yk_aekf = H @ xk
    yk_kf = H @ xk_fk
    if k % 1 == 0:
        y_aekf.append(yk_aekf)
        y_kf.append(yk_kf)


x_values = list(range(len(y_kf)))
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_noise, marker='.', color='blue')
plt.plot(x_values, y_real, marker='x', color='red')
plt.plot(x_values, y_kf, marker='o', color='green')

plt.title('KF')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.grid()

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_noise, marker='.', color='blue')
plt.plot(x_values, y_real, marker='x', color='red')
plt.plot(x_values, y_aekf, marker='o', color='green')

plt.title('AEKF')
plt.xlabel('Time step')
plt.ylabel('Error')
plt.grid()

plt.show()