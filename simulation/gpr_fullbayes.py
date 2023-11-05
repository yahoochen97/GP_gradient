import torch
import gpytorch
import numpy as np
import pandas as pd
import argparse
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from matplotlib import pyplot as plt
from utility.synthetic import generate_data
from gpytorch.priors import LogNormalPrior, UniformPrior

from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import GaussianLikelihood

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        self.x_covar_module = ScaleKernel(RBFKernel(active_dims=[0,1], ard_num_dims=2))
        self.it_covar_module = RBFKernel(active_dims=[2]) * ScaleKernel(RBFKernel(active_dims=[3]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.x_covar_module(x) + self.it_covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main(args):
    N = int(args["num_units"])
    T = int(args["num_times"])
    SEED = int(args["seed"])

    torch.manual_seed(12345)
    pyro.set_rng_seed(12345)
    
    # generate synthetic data
    train_x, train_y, true_grad = generate_data(n=N, T=T, SEED=SEED)

    likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
    model = GPModel(train_x, train_y, likelihood).double()

    hypers = {
        'likelihood.noise': torch.tensor(0.25),
        'x_covar_module.base_kernel.lengthscale': torch.tensor([1,1]),
        'x_covar_module.outputscale': torch.tensor(1.),
        'it_covar_module.kernels.0.lengthscale': torch.tensor(0.01),
        'it_covar_module.kernels.1.outputscale': torch.tensor(1.),
        'it_covar_module.kernels.1.base_kernel.lengthscale': torch.tensor(1.)
    }

    model.initialize(**hypers)
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    # register priors
    num_samples = 100
    warmup_steps = 100
    model.x_covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.1, 4.), "lengthscale")
    model.it_covar_module.kernels[1].base_kernel.register_prior("lengthscale_prior", UniformPrior(0.1, 4.), "lengthscale")
    model.x_covar_module.register_prior("outputscale_prior", UniformPrior(0.1, 4.), "outputscale")
    model.it_covar_module.kernels[1].register_prior("outputscale_prior", UniformPrior(0.1, 4.), "outputscale")
    model.it_covar_module.kernels[0].raw_lengthscale.requires_grad = False

    # mcmc sampling
    def pyro_model(x, y):
        with gpytorch.settings.fast_computations(False, False, False):
            sampled_model = model.pyro_sample_from_prior()
            output = sampled_model.likelihood(sampled_model(x))
            pyro.sample("obs", output, obs=y)
        return y

    nuts_kernel = NUTS(pyro_model)
    mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
    mcmc_run.run(train_x, train_y)

    # evaluate
    dx_mu = np.zeros((num_samples,train_x.size(0),2))
    dx_std = np.zeros((num_samples,train_x.size(0),2))
    evidence_gp = 0

    with gpytorch.settings.fast_pred_var():
        model.pyro_load_from_samples(mcmc_run.get_samples())
        model.eval()
        likelihood.eval()

        test_x = train_x.clone().detach().requires_grad_(True)
        observed_pred = model(test_x)
        loss = mll(observed_pred, train_y)
        evidence_gp += loss.sum()*train_x.size(0)

        for iter in range(num_samples):
            dy = torch.autograd.grad(observed_pred.mean[iter].sum(), test_x, retain_graph=True)[0]
            dx_mu[iter,:,0:2] = dy[:,0:2].numpy()

        n_samples = 100
        sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
        for iter in range(num_samples):
            one_sampled_pred = sampled_pred[:,iter,:].reshape(n_samples,-1)
            sampled_dx = torch.stack([torch.autograd.grad(pred.sum(),\
                                test_x, retain_graph=True)[0] for pred in one_sampled_pred])
            dx_std[iter, :,0:2] = sampled_dx[:,:,0:2].std(0).numpy()
    
    
    dx_std = ((dx_std**2).mean(0) + (dx_std**2).var(0)/num_samples)**0.5
    dx_mu = dx_mu.mean(0)
    evidence_gp /= num_samples
    print("model evidence: {:.3f}".format(evidence_gp))
    results = pd.DataFrame({"x1": train_x.numpy()[:,0],
                            "x2": train_x.numpy()[:,1],
                            "unit": train_x.numpy()[:,2],
                            "time": train_x.numpy()[:,3],
                            "y": train_y.numpy(),
                            "dx1": true_grad[:,0],
                            "dx1_mu": dx_mu[:,0],
                            "dx1_std": dx_std[:,0],
                            "dx2": true_grad[:,1],
                            "dx2_mu": dx_mu[:,1],
                            "dx2_std": dx_std[:,1]})
    
    result_filename = "./results/fullbayes_N" + str(N) + "_T" + str(T) + "_SEED" + str(SEED) + ".csv"
    results.to_csv(result_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='-N num_units -T num_times -s seed')
    parser.add_argument('-N','--num_units', help='number of units', required=True)
    parser.add_argument('-T','--num_times', help='number of times', required=True)
    parser.add_argument('-s','--seed', help='random seed', required=True)
    args = vars(parser.parse_args())
    main(args)