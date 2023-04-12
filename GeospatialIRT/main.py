import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from binomial_likelihood import BinomialLikelihood
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


# LOAD PACKAGES
import torch
import gpytorch

class BinomialGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(BinomialGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(1))
        # separate kernels for covariate, geospatial and time confounding
        self.x_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=[0,1], ard_num_dims=2))
        self.g_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=[2,3], ard_num_dims=2))
        self.t_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=[4]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        # sum over all covariances
        covar_x = self.x_covar_module(x)
        covar_g = self.g_covar_module(x)
        covar_t = self.t_covar_module(x)
        covar = covar_x + covar_t + covar_g
        return gpytorch.distributions.MultivariateNormal(mean_x, covar)

def main():
    '''
    Demonstration of dynamic Gaussian Process Item Response Theory
    with background characteristics and geospatial locations.
    '''

    # define data
    n = 100; # num of respondents
    T = 10; # num of time periods
    m = 10; # num of items in the battery
    train_x,train_y,f_min, f_max = generate_data(n,T,m)

    # initialize likelihood and model
    likelihood = BinomialLikelihood(m=m)
    model = BinomialGPModel(train_x=train_x).double()

    training_iterations = 100

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

    for i in range(training_iterations):
        # Zero backpropped gradients from previous iteration
        optimizer.zero_grad()
        # Get predictive output
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # plot the estimated latent f for first respondent
        test_x = train_x[0:T,:]
        # Get classification predictions
        observed_pred = likelihood(model(test_x))

        # Initialize fig and axes for plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.plot(test_x[:,4].numpy(), train_y[0:T].numpy(), 'k*')
        # Get the predicted labels
        pred_labels = observed_pred.mean.float()
        ax.plot(test_x[0:T,4].numpy(), pred_labels.numpy(), 'b')
        ax.legend(['Observed Data', 'Expected Responses'])
    plt.show()

    return 

def generate_data(n,T,m):
    np.random.seed(12345)
    x = np.zeros((n*T,2+2+1))
    y = np.zeros((n*T,))

    # background characteristics: x_1 ~ N(0,1), x_2 ~ N(0,1)
    x[:,0] = np.repeat(np.random.normal(loc=0.0, scale=1.0, size=(n,)),T)
    x[:,1] = np.repeat(np.random.normal(loc=0.0, scale=1.0, size=(n,)),T)

    # normalized geospatial locations: g_1 ~ Unif(0,1), g_2 ~ Unif(0,1)
    g = np.random.uniform(low=0.0, high=1.0, size=(n,2))
    x[:,2] = np.repeat(g[:,0],T)
    x[:,3] = np.repeat(g[:,1],T)

    # geospatial correlated noise: sigma(t) ~ N(0, K), K_{ij} = exp(-(g_i-g_j)^2/2), K_t = exp(-(t-t')^2/2) 
    kernel = RBF(length_scale = 1.0)
    # K = np.kron(kernel(g*2),kernel(np.arange(T).reshape(-1,1)/T*3))
    K = kernel(g*2)
    x[:,4] = np.tile(np.arange(T),n)
    g_effects = np.random.multivariate_normal(np.zeros((n,)), K)

    # normalized latent policy positions: 
    f = x[:,0]**2 + x[:,1] + x[:,0]*x[:,1] + (x[:,4]/T*2)**2 + np.repeat(g_effects, T)
    f_min = np.min(f)
    f_max = np.max(f)
    f = (f-f_min) / (f_max - f_min)
    
    # response model: y ~ binom(sigma_i,m), y=1,...,m
    for i in range(n*T):
        y[i] = np.random.binomial(m,f[i])

    return torch.from_numpy(x), torch.from_numpy(y), f_min, f_max
            
if __name__ == "__main__":
    main()
