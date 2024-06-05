# LOAD PACKAGES
import warnings
import torch
from gpytorch.distributions import base_distributions
from gpytorch.functions import log_normal_cdf
from gpytorch.likelihoods.likelihood import _OneDimensionalLikelihood

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import gpytorch

# implement BinomialLikelihood
class BinomialLikelihood(_OneDimensionalLikelihood):
    r"""
    Implements the Binomial likelihood for count data y between 1 and m. 
    The Binomial distribution is parameterized by :math:`m > 0`. 
    We can write the likelihood as:

    .. math::
        \begin{equation*}
            p(Y=y|f,m)=\phi(f)^y(1-\phi(f))^{(m-y)}
        \end{equation*}
    """

    def __init__(self, n_trials):
        super().__init__()
        self.n_trials = n_trials

    def forward(self, function_samples, **kwargs):
        output_probs = base_distributions.Normal(0, 1).cdf(function_samples)
        print(output_probs.size())
        return base_distributions.Binomial(total_count=self.n_trials, probs=output_probs)

    def log_marginal(self, observations, function_dist, *args, **kwargs):
        marginal = self.marginal(function_dist, *args, **kwargs)
        return marginal.log_prob(observations)

    def marginal(self, function_dist, **kwargs):
        mean = function_dist.mean
        var = function_dist.variance
        link = mean.div(torch.sqrt(1 + var))
        output_probs = base_distributions.Normal(0, 1).cdf(link)
        return base_distributions.Binomial(total_count=self.num_data, probs=output_probs)

    def expected_log_prob(self, observations, function_dist, *params, **kwargs):
        if torch.any(torch.logical_or(observations.le(-1), observations.ge(self.n_trials+1))):
            # Remove after 1.0
            warnings.warn(
                "BinomialLikelihood.expected_log_prob expects observations with labels in [0, m]. "
                "Observations <0 or >m are not allowed.",
                DeprecationWarning,
            )
        else:
            for i in range(observations.size(0)):
                observations[i] = torch.clamp(observations[i],0,self.n_trials[i])

        # Custom function here so we can use log_normal_cdf rather than Normal.cdf
        # This is going to be less prone to overflow errors
        log_prob_lambda = lambda function_samples: self.n_trials*log_normal_cdf(-function_samples) + \
                observations.mul(log_normal_cdf(function_samples)-log_normal_cdf(-function_samples))
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

# implement GP class
class BinomialGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, train_x):
        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution, learn_inducing_locations=False
        )
        super(BinomialGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.LinearMean(input_size=train_x.size(1))
        # ARD kernel for covariate, geospatial and time confounding
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.size(1)))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def transform_data(data):
    n = data.shape[0]
    x = np.zeros((n,11))
    y = np.zeros((n,))
    theta = np.zeros((n,))
    N = np.zeros((n,))
    x[:,0] = data["latitude"].to_numpy()
    x[:,1] = data["longitude"].to_numpy()
    x[:,2] = (data["gender"].to_numpy()=="Male")
    x[:,3] = (data["gender"].to_numpy()=="Female")
    x[:,4] = (data["gender"].to_numpy()=="Non-binary")
    x[:,5] = (data["race"].to_numpy()=="White")
    x[:,6] = (data["race"].to_numpy()=="Black")
    x[:,7] = (data["race"].to_numpy()=="Hispanic")
    x[:,8] = (data["race"].to_numpy()=="Asian")
    x[:,9] = (data["race"].to_numpy()=="Other")
    x[:,10] = data["year"].to_numpy()
    theta = data["theta"].to_numpy()
    y = data["Y"].to_numpy()
    N = data["n"].to_numpy()

    return torch.from_numpy(x).double(), torch.from_numpy(y).double(),\
            torch.from_numpy(N).double(), theta

def main():
    torch.manual_seed(0)

    # load data
    data = pd.read_csv("data.csv", index_col=0)

    # transform data
    train_x, train_y, train_N, true_theta = transform_data(data)

    # initialize likelihood and model
    likelihood = BinomialLikelihood(n_trials=train_N)
    model = BinomialGPModel(train_x=train_x).double()

    training_iterations = 200

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1) 

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

    print(model.covar_module.base_kernel.lengthscale)
    print(model.covar_module.outputscale)

    model.eval()
    likelihood.eval()

    with torch.no_grad():
        f_pred = model(train_x)
        mu = f_pred.mean.numpy()

    plt.scatter(true_theta,mu/np.std(mu))
    plt.xlabel("true theta")
    plt.ylabel("est theta")
    plt.show()

    results = pd.DataFrame({"true_theta": true_theta, "est_mean": mu/np.std(mu)})
    results['est_std'] = np.sqrt(f_pred.variance.numpy())/np.std(mu)
    results.to_csv("GPR_result.csv")

    lower = results['est_mean'] - 2*results['est_std']
    upper = results['est_mean'] + 2*results['est_std']
    print("Avg 95% Coverage: {:.3f}".format(np.mean(np.logical_and(lower<=true_theta, upper>=true_theta))))
    print("Avg RMSE: {:.3f}".format(np.sqrt(np.mean((true_theta-results['est_mean'])**2))))


if __name__ == "__main__":
    main()