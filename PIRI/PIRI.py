import torch
import pandas as pd
import gpytorch
from typing import Optional, Tuple
from matplotlib import pyplot as plt
from gpytorch.means import LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel

class ConstantVectorMeanWithGrad(gpytorch.means.mean.Mean):
    def __init__(self, d=1, prior=None, batch_shape=torch.Size(), **kwargs):
        super().__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector",\
                 parameter=torch.nn.Parameter(torch.zeros(*batch_shape, d)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constantvector")

    def forward(self, input):
        return self.constantvector[input.int().reshape((-1,)).tolist()]
    
    def grad(self, input):
        return torch.zero(input.size)

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
    
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dim=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = MaskMean(active_dims=1, \
                base_mean=ConstantVectorMeanWithGrad(d=train_x[:,1].unique().size()[0]))
        # year kernel * country kernel
        self.unit_covar_module = ScaleKernel(RBFKernel(active_dims=0)*RBFKernel(active_dims=1))
        self.x_covar_module = torch.nn.ModuleList([ScaleKernel(RBFKernel(\
            active_dims=(i))) for i in [6,7]])
        self.binary_covar_module = torch.nn.ModuleList([ScaleKernel(RBFKernel(\
            active_dims=(i))) for i in [3,4,5,8,9]])
        self.effect_covar_module = ScaleKernel(RBFKernel(active_dims=2))

    def forward(self, x):
        mean_x = self.mean_module(x)
        unit_covar_x = self.unit_covar_module(x)
        effect_covar_x = self.effect_covar_module(x)
        covar_x = unit_covar_x + effect_covar_x
        for i, _ in enumerate(self.x_covar_module):
            covar_x += self.x_covar_module[i](x)
        for i, _ in enumerate(self.binary_covar_module):
            covar_x += self.binary_covar_module[i](x)
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def load_PIRI_data():
    # read data
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
    y = torch.as_tensor(data.PIRI.to_numpy()).double()

    unit_means = torch.zeros(n,)
    for i in range(n):
        unit_means[i] = y[x[:,1]==i].mean()

    return x.double(), y.double(), unit_means.double(), data, countries, years

def main():
    train_x, train_y, unit_means, data, countries, years = load_PIRI_data()

    # define likelihood and derivativeExactGP model
    likelihood = GaussianLikelihood()
    model = GPModel(train_x, train_y, likelihood).double()

    # initialize model parameters
    hypers = {
        'mean_module.base_mean.constantvector': unit_means,
        'likelihood.noise_covar.noise': torch.tensor(0.25),
        'unit_covar_module.base_kernel.kernels.0.lengthscale': torch.tensor(4),
        'unit_covar_module.base_kernel.kernels.1.lengthscale': torch.tensor(0.01),
        'unit_covar_module.outputscale': torch.tensor(0.25),
        'x_covar_module.0.outputscale': torch.tensor(0.25),
        'x_covar_module.1.outputscale': torch.tensor(0.25),
        'binary_covar_module.0.base_kernel.lengthscale': torch.tensor(0.01),
        'binary_covar_module.1.base_kernel.lengthscale': torch.tensor(0.01),
        'binary_covar_module.2.base_kernel.lengthscale': torch.tensor(0.01),
        'binary_covar_module.3.base_kernel.lengthscale': torch.tensor(0.01),
        'binary_covar_module.4.base_kernel.lengthscale': torch.tensor(0.01),
        'effect_covar_module.base_kernel.lengthscale': torch.tensor(0.01),
        'effect_covar_module.outputscale': torch.tensor(0.25)
    }    

    model.initialize(**hypers)

    # train model
    model.train()
    likelihood.train()

    # freeze length scale in the country component in unit covar
    # freeze constant unit means
    all_params = set(model.parameters())
    final_params = list(all_params - \
             {model.unit_covar_module.base_kernel.kernels[1].raw_lengthscale, \
              model.mean_module.base_mean.constantvector, \
            #   model.x_covar_module[0].raw_outputscale,
            #   model.x_covar_module[1].raw_outputscale,
              model.binary_covar_module[0].base_kernel.raw_lengthscale,
              model.binary_covar_module[1].base_kernel.raw_lengthscale,
              model.binary_covar_module[2].base_kernel.raw_lengthscale,
              model.binary_covar_module[3].base_kernel.raw_lengthscale,
              model.binary_covar_module[4].base_kernel.raw_lengthscale})
            #   model.effect_covar_module.base_kernel.raw_lengthscale,
            #   model.effect_covar_module.raw_outputscale})
    optimizer = torch.optim.Adam(final_params, lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 50
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f '  % (
            i + 1, training_iter, loss.item()
        ))
        optimizer.step()
    
    # generate posterior of PIRI effects
    torch.save(model.state_dict(), "PIRI_GPR_model.pth")

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        out = likelihood(model(train_x))
        mu_f = out.mean
        V = out.covariance_matrix
        L = torch.linalg.cholesky(V, upper=False)

    # model.load_state_dict(full_model_state)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.unit_covar_module.outputscale = 0
        for i,_ in enumerate(model.x_covar_module):
            model.x_covar_module[i].outputscale = 0
        for i,_ in enumerate(model.binary_covar_module):
            model.binary_covar_module[i].outputscale = 0
        effect_covar = model(train_x).covariance_matrix

    # get posterior effect mean
    alpha = torch.linalg.solve(L.t(),torch.linalg.solve(L,train_y-mu_f))
    tmp = torch.linalg.solve(L, effect_covar)
    post_effect_mean = effect_covar @ alpha
    # get posterior effect covariance
    post_effect_covar = effect_covar - tmp.t() @ tmp

    effect_p = post_effect_mean[train_x[:,2]==1].mean().item()
    effect_n = post_effect_mean[train_x[:,2]==0].mean().item()
    effect = post_effect_mean[train_x[:,2]==1].mean() - post_effect_mean[train_x[:,2]==0].mean()
    effect_std = post_effect_covar.diag().mean().sqrt()
    BIC = (2+3+6+1)*torch.log(torch.tensor(train_x.size()[0])) + 2*loss*train_x.size()[0]
    print("effect: {:0.3f} +- {:0.3f}\n".format(effect, effect_std))
    print("model evidence: {:0.3f} \n".format(-loss*train_x.size()[0]))
    print("BIC: {:0.3f} \n".format(BIC))

    # get unit trend wo AIShame
    model.load_state_dict(torch.load('PIRI_GPR_model.pth'))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        model.effect_covar_module.outputscale = 0
        unit_covar = likelihood(model(train_x)).covariance_matrix

     # get posterior unit trend mean
    alpha = torch.linalg.solve(L.t(),torch.linalg.solve(L,train_y-mu_f))
    tmp = torch.linalg.solve(L, unit_covar)
    post_unit_mean = mu_f + unit_covar @ alpha
    # get posterior unit trend covariance
    post_unit_covar = unit_covar - tmp.t() @ tmp
    post_unit_covar = post_unit_covar.diag()

    # country-year post mu/std w/wo AIShame
    n = len(countries)
    m = len(years)
    results = pd.DataFrame(columns=['country',\
                     'year','PIRI','mu_D0','mu_D1', 'std','AIShame'])
    for i in range(n):
        for j in range(m):
            mask = (data.country==countries[i]) & (data.year==years[j])
            mask = mask.to_list()
            if sum(mask)>0:
                D = train_x[mask,2].item()
                mu = post_unit_mean[mask].item()
                results.loc[len(results)] = [countries[i],years[j], train_y[mask].item(),\
                    mu+effect_n, mu+effect_p, \
                    torch.sqrt(post_unit_covar[mask]).item(),D] 
    results.to_csv("./results/PIRI_GP_unit.csv", index=False)

if __name__ == "__main__":
    main()
