import torch
from sklearn.gaussian_process.kernels import RBF
import numpy as np

def generate_data(n=50, T=20, SEED=12345):
    np.random.seed(SEED)
    noise_sd = 0.1
    train_x_1 = np.random.normal(size=n*T)
    train_x_2 = np.random.normal(size=n*T)
    f_x = train_x_1**2 - train_x_1*train_x_2
    train_y = f_x + np.random.normal(size=n*T) * noise_sd

    df_x = np.zeros((n*T,2))
    df_x[:,0] = 2*train_x_1 - train_x_2
    df_x[:,1] = -train_x_1

    g_t = np.zeros((n,T))
    unit_kernel = RBF(length_scale=T/4)
    unit_cov = unit_kernel(X=np.arange(T).reshape((-1,1))) + 1e-6*np.eye(T)

    for i in range(n):
        g_t[i] = np.random.multivariate_normal(np.zeros((T,)),unit_cov)

    for i in range(n):
        train_y[(i*T):(i*T+T)] += g_t[i]

    train_x = np.zeros((n*T,2+1+1))
    # x1 and x2
    train_x[:,0] = train_x_1 # .repeat(T)
    train_x[:,1] = train_x_2 # .repeat(T)
    # unit index
    train_x[:,2] = np.arange(n).repeat(T)
    # time periods
    train_x[:,3] = np.tile(np.arange(T),n)
    
    return torch.from_numpy(train_x).double(), torch.from_numpy(train_y).double(), df_x
