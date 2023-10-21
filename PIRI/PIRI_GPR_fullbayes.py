# load gpytoch and other packages
import torch
import numpy as np
import pandas as pd
import gpytorch
import dill
from scipy.stats import norm
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from gpytorch.means import LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.priors import GammaPrior

# load pyro packages
import pyro
from pyro.infer.mcmc import NUTS, MCMC
import os
smoke_test = ('CI' in os.environ)
num_samples = 2 if smoke_test else 500
warmup_steps = 2 if smoke_test else 500
torch.set_default_dtype(torch.float64)

def load_PIRI_data():
    # read data
    # url = "http://raw.githubusercontent.com/yahoochen97/GP_gradient/main/hb_data_complete.csv"
    data = pd.read_csv("hb_data_complete.csv", index_col=[0])

    # all zero PIRI for new zealand and netherland
    data = data.loc[~data['country'].isin(['N-ZEAL','NETHERL'])]

    countries = sorted(data.country.unique())
    years = data.year.unique()
    n = len(countries)
    m = len(years)

    # build data
    country_dict = dict(zip(countries, range(n)))
    year_dict = dict(zip(years, range(m)))

    # x is:
    # 1: year number
    # 2: country id
    # 3: AIShame (treatment indicator)
    # 4: cat_rat
    # 5: ccpr_rat
    # 6: democratic
    # 7: log(gdppc)
    # 8: log(pop)
    # 9: Civilwar2
    # 10: War
    x = torch.zeros(data.shape[0], 10)
    x[:,0] = torch.as_tensor(list(map(year_dict.get, data.year)))
    x[:,1] = torch.as_tensor(list(map(country_dict.get, data.country)))
    x[:,2] = torch.as_tensor(data.AIShame.to_numpy())
    x[:,3] = torch.as_tensor(data.cat_rat.to_numpy())
    x[:,4] = torch.as_tensor(data.ccpr_rat.to_numpy())
    x[:,5] = torch.as_tensor(data.democratic.to_numpy())
    x[:,6] = torch.as_tensor(data.log_gdppc.to_numpy())
    x[:,7] = torch.as_tensor(data.log_pop.to_numpy())
    x[:,8] = torch.as_tensor(data.Civilwar2.to_numpy())
    x[:,9] = torch.as_tensor(data.War.to_numpy())
    # x[:,10] = torch.as_tensor(data.PIRI.to_numpy())
    y = torch.as_tensor(data.PIRILead1.to_numpy()).double()

    unit_means = torch.zeros(n,)
    for i in range(n):
        unit_means[i] = y[x[:,1]==i].mean()

    return x.double(), y.double(), unit_means.double(), data, countries, years

class ConstantVectorMean(gpytorch.means.mean.Mean):
    def __init__(self, d=1, prior=None, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector",\
                 parameter=torch.nn.Parameter(torch.zeros(*batch_shape, d)))
        if prior is not None:
            self.register_prior("constantvector_prior", prior, "constantvector")

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
    

train_x, train_y, unit_means, data, countries, years = load_PIRI_data()

# model specification: PIRI gp model with unit trends
# x_it : AIShame + cat_rat + ccpr_rat 
#            + democratic + log(gdppc) + log(pop) 
#            + Civilwar2 + War 
# y_i(t) ~ u_i(t) + f(x_{it}) + ε
# f(x_{it}) ~ GP(w^Tx, K_x)
# u_i(t) ~ GP(b_i, K_t)
import statsmodels.formula.api as sm

lm = sm.ols('PIRILead1 ~ AIShame  + cat_rat + ccpr_rat \
            + democratic + log_gdppc + log_pop \
            + Civilwar2 + War + C(year) + C(country) + PIRI', data).fit()

coefs = lm.params.to_dict()
covariate_names = ["AIShame" ,"cat_rat" , "ccpr_rat",
           "democratic",  "log_gdppc", "log_pop",
            "Civilwar2", "War"]
x_weights = list(map(coefs.get, covariate_names))

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood

        # constant country-level mean; fix; no prior
        # self.mean_module = MaskMean(active_dims=1, \
        #        base_mean=ConstantVectorMean(d=train_x[:,1].unique().size()[0]))
        
        # linear mean for continuous and binary covariates
        # self.x_mean_module = MaskMean(active_dims=[2,3,4,5,6,7,8,9], base_mean=LinearMean(input_size=8, bias=False))
        self.mean_module = gpytorch.means.ZeroMean()
        # unit level trend: year kernel * country kernel
        self.unit_covar_module = ScaleKernel(RBFKernel(active_dims=0)*RBFKernel(active_dims=1))
        self.x_covar_module = ScaleKernel(RBFKernel(active_dims=[2,3,4,5,6,7,8,9],ard_num_dims=8))

    def forward(self, x):
        mean_x = self.mean_module(x) # + self.x_mean_module(x)
        unit_covar_x = self.unit_covar_module(x)
        covar_x = unit_covar_x + self.x_covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = GPModel(train_x, train_y, likelihood).double()

# initialize model parameters
hypers = {
    # 'mean_module.base_mean.constantvector': unit_means,
    # 'x_mean_module.base_mean.weights': torch.tensor(x_weights),
    'likelihood.noise_covar.noise': torch.tensor(0.5),
    'unit_covar_module.base_kernel.kernels.0.lengthscale': torch.tensor(6.),
    'unit_covar_module.base_kernel.kernels.1.lengthscale': torch.tensor(0.01),
    'unit_covar_module.outputscale': torch.tensor(4.),
    'x_covar_module.outputscale': torch.tensor(1.)
}    

model = model.initialize(**hypers)

# fix constant prior mean
# model.mean_module.base_mean.constantvector.requires_grad = False
model.unit_covar_module.base_kernel.kernels[1].raw_lengthscale.requires_grad = False

# register priors
model.unit_covar_module.register_prior("outputscale_prior", GammaPrior(1.0, 1.), "outputscale")
model.unit_covar_module.base_kernel.kernels[0].register_prior("lengthscale_prior", GammaPrior(2., 1.), "lengthscale")
model.x_covar_module.base_kernel.register_prior("lengthscale_prior", GammaPrior(1., 1.), "lengthscale")
model.x_covar_module.register_prior("outputscale_prior", GammaPrior(1., 1.), "outputscale")
likelihood.register_prior("noise_prior", GammaPrior(1., 1.0), "noise")

# Initialize with MAP
model.train()
likelihood.train()
torch.manual_seed(12345)

# freeze length scale in the country component in unit covar
# freeze constant unit means
all_params = set(model.parameters())
final_params = list(all_params - \
            {model.unit_covar_module.base_kernel.kernels[1].raw_lengthscale})
#, \
#            model.mean_module.base_mean.constantvector})
optimizer = torch.optim.Adam(final_params, lr=0.05)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x)
    loss = -mll(output, train_y)
    loss.backward()
    if i % 20 == 0:
        print('Iter %d/%d - Loss: %.3f '  % (
            i , training_iter, loss.item()
        ))
    optimizer.step()

# train model
model.train()
likelihood.train()

pyro.set_rng_seed(12345)

def pyro_model(x, y):
    sampled_model = model.pyro_sample_from_prior()
    output = sampled_model.likelihood(sampled_model(x))
    pyro.sample("obs", output, obs=y)

# print("loading mcmc run from disk...")
# with open('./results/PIRI_GPR_fullbayes_500.pkl', 'rb') as f:
#     mcmc_run = dill.load(f)

nuts_kernel = NUTS(pyro_model)
mcmc_run = MCMC(nuts_kernel, num_samples=num_samples,\
            warmup_steps=warmup_steps, disable_progbar=smoke_test,\
            num_chains=1)
mcmc_run.run(train_x, train_y)

with open('./results/PIRI_GPR_fullbayes.pkl', 'wb') as f:
    dill.dump(mcmc_run, f)

import pandas as pd
results = pd.DataFrame(columns=['v_name','n_eff','r_hat'])

for k,v in mcmc_run.diagnostics().items():
    if k=='divergences' or k=="acceptance rate": continue
    if k=="x_covar_module.base_kernel.lengthscale_prior":
        for i in range(v['n_eff'].shape[1]):
            results = results.append({'v_name':k+"_"+str(i), 'n_eff': v['n_eff'].numpy()[0,i],\
                             'r_hat': v['r_hat'].numpy()[0,i]}, ignore_index=True)
    else:
        results = results.append({'v_name':k, 'n_eff': v['n_eff'].numpy(), \
                              'r_hat': v['r_hat'].numpy()}, ignore_index=True)

print(results)

# iterate over each mcmc sample
est_means = np.zeros((num_samples, len(covariate_names)))
est_stds = np.zeros((num_samples, len(covariate_names)))
samples = mcmc_run.get_samples()
for iter in range(num_samples):
    one_sample = {}
    for k,v in samples.items():
        one_sample[k] = v[iter]
    model.pyro_load_from_samples(one_sample)
    model.eval()
    likelihood.eval()

    df_std = np.zeros((train_x.size(0),train_x.size(1)))
    x_grad = np.zeros((train_x.size(0),train_x.size(1)))

    # number of empirically sample 
    n_samples = 100
    sampled_dydtest_x = np.zeros((n_samples, train_x.size(0),train_x.size(1)))

    # we proceed in small batches of size 100 for speed up
    print(iter)

    for i in range(train_x.size(0)//100):
        with gpytorch.settings.fast_pred_var():
            test_x = train_x[(i*100):(i*100+100)].clone().detach().requires_grad_(True)
            observed_pred = model(test_x)
            dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
            x_grad[(i*100):(i*100+100)] = dydtest_x

            sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
            sampled_dydtest_x[:,(i*100):(i*100+100),:] = torch.stack([torch.autograd.grad(pred.sum(), \
                                        test_x, retain_graph=True)[0] for pred in sampled_pred])
            
    # last 100 rows
    with gpytorch.settings.fast_pred_var():
        test_x = train_x[(100*i+100):].clone().detach().requires_grad_(True)
        observed_pred = model(test_x)
        dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
        x_grad[(100*i+100):] = dydtest_x

        sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
        sampled_dydtest_x[:,(100*i+100):,:] = torch.stack([torch.autograd.grad(pred.sum(),\
                                        test_x, retain_graph=True)[0] for pred in sampled_pred])
        

    est_std = np.sqrt(sampled_dydtest_x.mean(1).std(0)**2 + \
                    sampled_dydtest_x.std(1).mean(0)**2).round(decimals=5)
    est_stds[iter] = est_std[2:10]
    est_means[iter] = x_grad.mean(axis=0)[2:10]

# print marginalized results
results = pd.DataFrame({"x": covariate_names, \
                    'est_mean': est_means.mean(axis=0),
                    'est_std': np.sqrt(np.var(est_means.mean(axis=0)) + np.power(est_stds, 2).mean(axis=0))})
results["t"] = results['est_mean'].values/results['est_std'].values
results["pvalue"] = 1 - norm.cdf(np.abs(results["t"].values))
results.to_csv("./results/PIRI_GPR_fullbayes.csv")
print(results)