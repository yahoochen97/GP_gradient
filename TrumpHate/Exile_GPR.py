# load gpytoch and other packages
import torch
import numpy as np
import pandas as pd
import gpytorch
from scipy.stats import norm
from matplotlib import pyplot as plt
from gpytorch.means import ZeroMean, LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from datetime import datetime

from gpytorch.means import Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from typing import Optional, Tuple
from torch.utils.data import TensorDataset, DataLoader

torch.set_default_dtype(torch.float64)
torch.manual_seed(12345)

num_inducing = 5000
batch_size = 256
num_epochs = 100

def diff_month(d1, d2):
    d1 = datetime.strptime(d1,"%Y-%m-%d")
    d2 = datetime.strptime(d2,"%Y-%m-%d")
    return (d1.year - d2.year) * 12 + d1.month - d2.month

def to_month(d1):
    return datetime(2013 + int(d1 / 12), ((1 +d1) % 12) + 1, 1)

class ConstantVectorMean(gpytorch.means.mean.Mean):
    def __init__(self, d=1, prior=None, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector",\
                 parameter=torch.nn.Parameter(torch.zeros(*batch_shape, d)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constantvector")

    def forward(self, input):
        return self.constantvector[input.int().reshape((-1,)).tolist()]
    
class MaskMean(gpytorch.means.mean.Mean):
    def __init__(
        self,
        base_mean: gpytorch.means.mean.Mean,
        active_dims: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ):
        super().__init__()
        if active_dims is not None and not torch.is_tensor(active_dims):
            active_dims = torch.tensor(active_dims, dtype=torch.long)
        self.active_dims = active_dims
        self.base_mean = base_mean
    
    def forward(self, x, **params):
        return self.base_mean.forward(x.index_select(-1, self.active_dims), **params)

class GPModel(ApproximateGP):
    def __init__(self, inducing_points, unit_num):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=False)
        super(GPModel, self).__init__(variational_strategy)

        # linear mean
        self.mean_module = LinearMean(input_size=(3), bias=True)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=(3), active_dims=[2,3,4]))
        self.t_covar_module = ScaleKernel(RBFKernel(active_dims=[0])*RBFKernel(active_dims=[1]))

    def forward(self, x):
        mean_x = self.mean_module(x[:,2:5]) 
        covar_x =  self.covar_module(x) + self.t_covar_module(x) 
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main(Y_name):
    # read data
    data = pd.read_csv("./data/exile.csv")
    data = data[[Y_name, "tweeted_exile", "month","num_tweets", "actor.id"]]

    # xs: unit id, month, log_num_tweets, tweeted_exile, month *  tweeted_exile
    xs = data.month.apply(lambda x: diff_month(x,"2013-01-01"))
    xs = torch.tensor(np.array([data["actor.id"].astype('category').cat.codes.values.reshape((-1,)),\
                        xs.values.reshape((-1,)),
                        np.log(data.num_tweets.values+1).reshape((-1,)), \
                        data['tweeted_exile'].values.reshape((-1,))]).T)
    xs = torch.cat((xs, (xs[:, 1] * xs[:, -1]).reshape(-1,1)), dim=1)
    ys = torch.tensor(data[Y_name].values).double()

    # define inducing points and learn
    inducing_points = xs[np.random.choice(xs.size(0),num_inducing,replace=False),:]
    # inducing_points = xs[xs[:,1] % 10==0]
    model = GPModel(inducing_points=inducing_points, unit_num=xs[:,0].unique().size()[0]).double()
    likelihood = GaussianLikelihood().double()

    train_dataset = TensorDataset(xs, ys)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    hypers = {
        'mean_module.bias': torch.mean(ys),
        'mean_module.weights': torch.tensor([0, 5, 0]),
        'covar_module.outputscale': 1,
        'covar_module.base_kernel.lengthscale': torch.std(xs[:,2:5],axis=0),
        't_covar_module.base_kernel.kernels.1.lengthscale': torch.tensor([36]),
        't_covar_module.outputscale': 9
    }    

    model = model.initialize(**hypers)

    # initialize model parameters
    model.mean_module.bias.requires_grad_(False)
    model.t_covar_module.base_kernel.kernels[0].raw_lengthscale.requires_grad_(False)
    model.t_covar_module.base_kernel.kernels[1].raw_lengthscale.requires_grad_(False)
    model.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
    model.t_covar_module.base_kernel.kernels[0].lengthscale = 0.01
    likelihood.noise = 9.

    # train model
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': list(set(model.parameters()) \
                    - {model.t_covar_module.base_kernel.kernels[0].raw_lengthscale,\
                    })},
        {'params': likelihood.parameters()},
    ], lr=0.1)

    # "Loss" for GPs
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=ys.size(0))

    for i in range(num_epochs):
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            if j % 40 == 0:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))

    # set model and likelihood to evaluation mode
    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out = model(xs)
        mll.combine_terms = True
        loss = -mll(out, ys)
        mu_f = out.mean.numpy()
        lower, upper = out.confidence_region()

    to_unit = dict(enumerate(data["actor.id"].astype('category').cat.categories))

    # store results
    results = pd.DataFrame({"gpr_mean":mu_f})
    results['true_y'] = ys
    results['gpr_lwr'] = lower
    results['gpr_upr'] = upper
    results['month'] = np.array([to_month(x) for x in xs[:,1].numpy().astype(int)])
    results['unit'] = np.array([to_unit[x] for x in xs[:,0].numpy().astype(int)])
    results['exile'] = xs[:,3].numpy().astype(int)

    test_x0 = xs.clone().detach().requires_grad_(False)
    test_x0[:,3:5] = 0

    # in eval mode the forward() function returns posterioir
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out0 = model(test_x0)
        lower, upper = out0.confidence_region()

    results['cf'] = out0.mean.numpy()
    results['cf_lower'] = lower
    results['cf_upper'] = upper

    if Y_name == "perc_harsh_criticism":
        abbr = "crit"
    else:
        abbr = "repr"
    results.to_csv("./results/exile_{}_fitted_gpr.csv".format(abbr),index=False) #save to file

    model.eval()
    likelihood.eval()

    # copy training tesnor to test tensors and set exile to 1 and 0
    test_x1 = xs.clone().detach().requires_grad_(False)
    test_x1[:,3] = 1
    test_x1[:,4] = test_x1[:,1]
    test_x0 = xs.clone().detach().requires_grad_(False)
    test_x0[:,3:5] = 0

    # in eval mode the forward() function returns posterioir
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out = model(xs)
        mll.combine_terms = False
        loss, _ , _ = mll(out, ys)
        loss = -loss*out.event_shape[0]
        out1 = model(test_x1)
        out0 = model(test_x0)

    # compute ATE and its uncertainty
    effect = out1.mean.numpy()[xs[:,3]==1].mean() - out0.mean.numpy()[xs[:,3]==1].mean()
    effect_std = np.sqrt((out1.variance.detach().numpy()[xs[:,3]==1].mean()\
                        +out0.variance.detach().numpy()[xs[:,3]==1].mean()))
    BIC = (3+2+1)*\
        torch.log(torch.tensor(xs.size()[0])) + 2*loss # *xs.size(0)/batch_size
    print("ATE: {:0.3f} +- {:0.3f}\n".format(effect, effect_std))
    print("model evidence: {:0.3f} \n".format(-loss))
    print("BIC: {:0.3f} \n".format(BIC))

    print(likelihood.noise)
    print(model.t_covar_module.outputscale)
    print(model.t_covar_module.base_kernel.kernels[1].lengthscale)
    print(model.covar_module.outputscale)
    print(model.covar_module.base_kernel.lengthscale)


    model.eval()
    likelihood.eval()

    # number of empirically sample 
    n_samples = 100
    x_grad = np.zeros((xs.size(0),xs.size(1)))
    sampled_dydtest_x = np.zeros((n_samples, xs.size(0),xs.size(1)))

    # we proceed in small batches of size 100 for speed up
    for i in range(xs.size(0)//100):
        with gpytorch.settings.fast_pred_var():
            test_x = xs[(i*100):(i*100+100)].clone().detach().requires_grad_(True)
            observed_pred = likelihood(model(test_x))
            dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
            x_grad[(i*100):(i*100+100)] = dydtest_x

            sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
            sampled_dydtest_x[:,(i*100):(i*100+100),:] = torch.stack([torch.autograd.grad(pred.sum(), \
                                        test_x, retain_graph=True)[0] for pred in sampled_pred])
            
    # last 100 rows
    with gpytorch.settings.fast_pred_var():
        test_x = xs[(100*i+100):].clone().detach().requires_grad_(True)
        observed_pred = likelihood(model(test_x))
        dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
        x_grad[(100*i+100):] = dydtest_x

        sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
        sampled_dydtest_x[:,(100*i+100):,:] = torch.stack([torch.autograd.grad(pred.sum(),\
                                        test_x, retain_graph=True)[0] for pred in sampled_pred])
        
    est_std = np.sqrt(sampled_dydtest_x.mean(1).std(0)**2 + \
                  sampled_dydtest_x.std(1).mean(0)**2).round(decimals=5)

    covariate_names = ["log_num_tweets"]
    results = pd.DataFrame({"x": covariate_names, \
                            'est_mean': x_grad.mean(axis=0)[2],
                            'est_std': est_std[2]})
    results["t"] = results['est_mean'].values/results['est_std'].values
    results["pvalue"] = 1 - norm.cdf(np.abs(results["t"].values))
    print(results)

Y_names = ["perc_harsh_criticism", "perc_repression"]

for Y_name in Y_names:
    main(Y_name)