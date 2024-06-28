# load gpytoch and other packages
import torch
import numpy as np
import pandas as pd
import gpytorch
from scipy.stats import norm
from matplotlib import pyplot as plt
from gpytorch.means import LinearMean, ZeroMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, CosineKernel
from datetime import datetime

# read data
data = pd.read_csv("./data/white_nationalist_random.csv")

# format date as datatime
# and turn date into time index
xs = data.date
xs = xs.apply(lambda x: (datetime.strptime(x,"%Y-%m-%d")-datetime.strptime(xs[0],"%Y-%m-%d")).days)
ys_scale = 1e6
ys = torch.tensor(data.norm_hate.values).double() * ys_scale

# xs: time, election indicator, time * election indicator
xs = torch.tensor(np.array([xs.values, data.election.values,
                            xs.values*data.election.values]).T).double()
# xs = xs[:,0:2]
election_day_index = np.where(xs[:,1])[0][0]


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        # constant country-level mean
        self.mean_module = LinearMean(input_size=train_x.shape[1], bias=False)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
import statsmodels.formula.api as sm

data["time_index"] = xs[:,0].numpy()
lm = sm.ols('norm_hate ~ time_index + election + time_index:election - 1', data).fit()
coefs = lm.params.to_dict()
covariate_names = ["time_index" ,"election" , "time_index:election"]
x_weights = list(map(coefs.get, covariate_names))
print(lm.summary())


torch.set_default_dtype(torch.float64)
torch.manual_seed(12345)

likelihood = GaussianLikelihood()
model = GPModel(xs, ys, likelihood).double()

# initialize model parameters
hypers = {
    'mean_module.weights': torch.tensor(x_weights), #
    'covar_module.outputscale': torch.var(ys),
    'covar_module.base_kernel.lengthscale': torch.tensor([90.,1, 90]),
    'likelihood.noise': 1,
}    

model = model.initialize(**hypers)

# train model
model.train()
likelihood.train()

all_params = set(model.parameters())
model.covar_module.base_kernel.raw_lengthscale.requires_grad = False
# model.covar_module.raw_outputscale.requires_grad = False
# model.mean_module.weights.requires_grad = False
optimizer = torch.optim.Adam(all_params, lr=0.05)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 500
losses = []
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(xs)
    loss = -mll(output, ys)
    loss.backward()
    if i % 100 == 0:
        print('Iter %d/%d - Loss: %.3f '  % (
            i , training_iter, loss.item()
        ))
    losses.append(loss.item())
    optimizer.step()

# set model and likelihood to evaluation mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    out = model(xs)
    loss = -mll(output, ys)-np.log(ys_scale)
    mu_f = out.mean.numpy()
    lower, upper = out.confidence_region()

# store results
results = pd.DataFrame({"gpr_mean":mu_f/ys_scale})
results['true_y'] = ys/ys_scale
results['gpr_lwr'] = lower/ys_scale
results['gpr_upr'] = upper/ys_scale
results['day'] = xs[:,0].numpy().astype(int)
results.to_csv("./results/TrumpHate_fitted_gpr.csv",index=False) #save to file

f, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.vlines(xs[:,0].numpy(), 0*ys.numpy(), ys.numpy()/ys_scale, colors='k')
ax.plot(xs[:,0].numpy(), out.mean.numpy()/ys_scale, 'b')
ax.fill_between(xs[:,0].numpy(), lower.numpy()/ys_scale, upper.numpy()/ys_scale, alpha=0.5)
ax.axvline(x=election_day_index, ls="--")
ax.text(election_day_index-80, ys.numpy().max()/ys_scale, "Election Day", rotation=0, verticalalignment='center')
ax.legend(['Data', 'GPR mean', '95% CI'], frameon=False)
ax.set_xlabel("Date")
ax.set_ylabel("Daily Proportion of Tweets")
ax.set_xticks(np.arange(xs[:,0].min().data,xs[:,0].max().data,120))
ax.set_xticklabels(["06-2015","10-2015","02-2016","06-2016","10-2016","02-2017","06-2017"])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("./results/trumphate.pdf", dpi=100)


print(-loss*xs.size(0))
print(likelihood.noise)
print(model.covar_module.outputscale)
print(model.covar_module.base_kernel.lengthscale)



# copy training tesnor to test tensors and set election to 1 and 0
test_x1 = xs.clone().detach().requires_grad_(False)
test_x1[:,1] = 1
test_x1[:,2] = test_x1[:,0]
test_x0 = xs.clone().detach().requires_grad_(False)
test_x0[:,1] = 0
test_x0[:,2] = 0

# in eval mode the forward() function returns posterioir
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    out1 = model(test_x1)
    out0 = model(test_x0)

# compute ATE and its uncertainty
mask1 = (xs[:,0] == election_day_index + 1) & (xs[:,1]==1)
mask0 = (xs[:,0] == election_day_index) & (xs[:,1]==1)
effect = out1.mean.numpy()[mask1][0]-out0.mean.numpy()[mask0][0]
effect_std = np.sqrt((out1.variance.detach().numpy()[mask1]\
                      + out0.variance.detach().numpy()[mask0]))[0]

print("instaneous shift on Election Day: {:.2E} +- {:.2E}\n".format(effect/ys_scale, effect_std/ys_scale))
BIC = (3+1+1)*torch.log(torch.tensor(xs.size(0))) + 2*loss*xs.size()[0]
print(norm.cdf(-np.abs(effect/effect_std)))
print("log lik: {:4.4f} \n".format(-loss.numpy()*xs.size(0)))
print("BIC: {:0.3f} \n".format(BIC))


# compute ATE and its uncertainty
mask1 = (xs[:,0] >= election_day_index ) & (xs[:,1]==1)
effect = out1.mean.numpy()[mask1].mean()-out0.mean.numpy()[mask1].mean()
effect_std = np.sqrt((out1.variance.detach().numpy()[mask1].mean()\
                      + out0.variance.detach().numpy()[mask1].mean()))

print("ATE: {:.2E} +- {:.2E}\n".format(effect/ys_scale, effect_std/ys_scale))


# set model and likelihood to evaluation mode
model.eval()
likelihood.eval()
model.covar_module.outputscale = 0.1

xss = xs.clone().detach().requires_grad_(False)
xss[:,1] = 0
xss[:,2] = 0
with torch.no_grad():
    out0 = model(xss)
    mu_f0 = out0.mean.numpy()
    lower0, upper0 = out0.confidence_region()

f, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.vlines(xs[:,0].numpy(), 0*ys.numpy(), ys.numpy()/ys_scale, color='k')
ax.plot(xs[:,0].numpy(), out.mean.numpy()/ys_scale, 'b')
ax.fill_between(xs[:,0].numpy(), lower.numpy()/ys_scale, upper.numpy()/ys_scale, alpha=0.5)

ax.plot(xs[:,0].numpy()[xs[:,1]==1], out0.mean.numpy()[xs[:,1]==1]/ys_scale, 'g')
ax.fill_between(xs[:,0].numpy()[xs[:,1]==1], lower0.numpy()[xs[:,1]==1]/ys_scale, \
                upper0.numpy()[xs[:,1]==1]/ys_scale, alpha=0.5)

ax.axvline(x=election_day_index, ls="--")
ax.text(election_day_index-80, ys.numpy().max()/ys_scale, "Election Day", rotation=0, verticalalignment='center')
ax.text(election_day_index-200, ys.numpy().max()/ys_scale - 0.2, "ATE: {:.2E} +- {:.2E}\n".format(effect/ys_scale, effect_std/ys_scale), \
        rotation=0, verticalalignment='center', color='gray')

ax.legend(['Data', 'Factual mean', 'Factual 95% CI', 'Counterfactual mean',\
            'Counterfactual 95% CI'], loc='best', frameon=False)
ax.set_xlabel("Date")
ax.set_ylabel("Daily Proportion of Tweets")
ax.set_xticks(np.arange(xs[:,0].min().data,xs[:,0].max().data,120))
ax.set_xticklabels(["06-2015","10-2015","02-2016","06-2016","10-2016","02-2017","06-2017"])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig("./results/trumphate_est.pdf", dpi=100)



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
        observed_pred = model(test_x)
        dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
        x_grad[(i*100):(i*100+100)] = dydtest_x

        sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
        sampled_dydtest_x[:,(i*100):(i*100+100),:] = torch.stack([torch.autograd.grad(pred.sum(), \
                                    test_x, retain_graph=True)[0] for pred in sampled_pred])
        
# last 100 rows
with gpytorch.settings.fast_pred_var():
    test_x = xs[(100*i+100):].clone().detach().requires_grad_(True)
    observed_pred = model(test_x)
    dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x, retain_graph=True)[0]
    x_grad[(100*i+100):] = dydtest_x

    sampled_pred = observed_pred.rsample(torch.Size([n_samples]))
    sampled_dydtest_x[:,(100*i+100):,:] = torch.stack([torch.autograd.grad(pred.sum(),\
                                     test_x, retain_graph=True)[0] for pred in sampled_pred])

est_std = np.sqrt(sampled_dydtest_x.var((0,1))/xs.size(0)/n_samples).round(decimals=16)
covariate_names = ["beta1", "beta2", "beta3"]
results = pd.DataFrame({"x": covariate_names, \
                        'est_mean': x_grad.mean(axis=0)[[0,1,2]]/ys_scale,
                        'est_std': est_std[[0,1,2]]/ys_scale})
results["t"] = results['est_mean'].values/results['est_std'].values
results["pvalue"] = norm.cdf(-np.abs(results["t"].values))
print(results)