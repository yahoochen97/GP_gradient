import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.means import LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from models.DerivativeExactGP import DerivativeExactGP, ConstantVectorMean, myIndicatorKernel


# class GPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = LinearMean(input_size=train_x[0].shape[1], bias=False)
#         self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x[0].shape[1]))
#         self.i_mean_module =  ConstantVectorMean(d=train_x[1].unique().shape[0])
#         self.t_mean_module = ConstantVectorMean(d=train_x[2].unique().shape[0])
#         self.i_covar_module = ScaleKernel(myIndicatorKernel(train_x[1].unique().shape[0]))
#         self.t_covar_module = ScaleKernel(myIndicatorKernel(train_x[2].unique().shape[0]))

#     def forward(self, x, i, t):
#         mean_x = self.mean_module(x) + self.i_mean_module(i) + self.t_mean_module(t)
#         covar_x = self.covar_module(x) + self.i_covar_module(i) + self.t_covar_module(t)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def generate_data(D, n, T):
    torch.random.manual_seed(12345)
    noise_sd = 0.1
    soboleng = torch.quasirandom.SobolEngine(dimension=D)
    train_x = 8*soboleng.draw(n)-4
    train_i = torch.zeros((n*T,1))
    train_t = torch.zeros((n*T,1))
    train_x_all = torch.zeros((n*T,D))
    train_y = torch.randn(n*T) * noise_sd

    i_effects = torch.randn(n)
    t_effects = torch.randn(T)

    ARD_lss = 1 + torch.arange(D)

    for i in range(n):
        for t in range(T):
            train_x_all[t+i*T,] = train_x[i,]
            train_i[t+i*T,] = i
            train_t[t+i*T,] = t
            for k in range(D):
                train_y[t+i*T,] += torch.sin(train_x[i,k]*ARD_lss[k])/ARD_lss[k]
            train_y[t+i*T,] += i_effects[i] + t_effects[t] 
    
    # model = GPModel((train_x_all, train_i, train_t), train_y, GaussianLikelihood())

    # with torch.no_grad(), gpytorch.settings.prior_mode(True):
    #     train_y = model(train_x_all,train_i,train_t).sample()

    return train_x_all, train_i, train_t, train_y, ARD_lss

def main():
    D = 2
    n = 500
    T = 1
    train_x, train_i, train_t, train_y, ARD_lss = generate_data(D, n, T)
    train_i = train_i.reshape((-1,))
    train_t = train_t.reshape((-1,))

    likelihood = GaussianLikelihood(noise_prior=gpytorch.priors.UniformPrior(0.1, 0.5),\
        noise_constraint=gpytorch.constraints.GreaterThan(1e-1))
    model = DerivativeExactGP(n=n, T=T, D=D, likelihood=likelihood,\
        outputscale_hyperprior=gpytorch.priors.UniformPrior(0.5, 2.0),\
        outputscale_constraint=gpytorch.constraints.GreaterThan(0.5),\
        lengthscale_hyperprior=gpytorch.priors.UniformPrior(0.5, 1.5),
        lengthscale_constraint=gpytorch.constraints.LessThan(1.5))
    model.append_train_data((train_x,train_i,train_t), train_y)

    hypers = {
        'likelihood.noise_covar.noise': torch.tensor(0.1),
        'covar_module.base_kernel.lengthscale': torch.tensor([1,1]),
        'covar_module.outputscale': torch.tensor(1.),
        'i_covar_module.outputscale': torch.tensor(1.),
        't_covar_module.outputscale': torch.tensor(1.),
    }

    model.initialize(**hypers)

    # train
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 20
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x,train_i,train_t)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f  slope: %.3f   lengthscale1: %.3f  lengthscale2: %.3f    outputscale: %.3f  noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.mean_module.weights[0].item(),
            model.covar_module.base_kernel.lengthscale[0,0].item(),
            model.covar_module.base_kernel.lengthscale[0,1].item(),
            model.covar_module.outputscale.item(),
            likelihood.noise.item()
        ))
        optimizer.step()

    print(model.covar_module.base_kernel.lengthscale.tolist())

    # evaluate
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(train_x, train_i, train_t))
    
    with torch.no_grad():
        mean_d, variance_d = model.posterior_derivative(train_x)
        variance_d = torch.diagonal(variance_d, dim1=1,dim2=2)

    # X = torch.autograd.Variable(train_x, requires_grad=True)
    # def f(X):
    #     return likelihood(model(X,train_i,train_t)).mean.sum()
    # mean_d  = torch.autograd.functional.jacobian(f, X)
    # y = likelihood(model(X,train_i,train_t)).mean.sum()
    # y.backward()
    # mean_d = X.grad
        
    with torch.no_grad():
        # Initialize plot
        _, ax = plt.subplots(nrows=2,ncols=1,figsize=(6, 8))

        # ax[0].plot(train_x[:,0].numpy(), train_y.numpy(), '--k',label='data')
        sorted, idx = torch.sort(train_x[:,0])
        ax[0].plot(sorted.numpy(), torch.cos(sorted*ARD_lss[0]).numpy(), '--k',label='true grad')
        ax[0].errorbar(sorted.numpy(), mean_d[idx,0].numpy(), yerr=2*variance_d[idx,0].numpy()**0.5,\
             marker='s', mfc='red', mec='black', ms=2, mew=2,label='estimated grad')

        ax[0].legend()
        ax[0].set_title('Estimated partial gradient of x1')

        # ax[1].plot(train_x[:,4].numpy(), train_y.numpy(), '*k',label='data')
        sorted, idx = torch.sort(train_x[:,1])
        ax[1].plot(sorted.numpy(), torch.cos(sorted*ARD_lss[1]).numpy(), '--k',label='true grad')
        
        ax[1].errorbar(sorted.numpy(), mean_d[idx,1].numpy(), yerr=2*variance_d[idx,1].numpy()**0.5,\
             marker='s', mfc='red', mec='black', ms=2, mew=2,label='estimated grad')
        ax[1].legend()
        ax[1].set_title('Estimated partial gradient of x2')
        # ax[1].set_ylim([-6, 6])
        # ax[1].legend(['Gradient', 'Mean', 'Confidence'])
        # ax[1].set_title('Estimated gradient')
        plt.savefig("2FE_gradient_T" + str(T) + "_n" + str(n) + ".pdf",dpi=300)
        plt.show()

if __name__ == "__main__":
    main()

    # TODO: dynamic time confounding
    # TODO: Spatial confounding (same ls for long/latitude)
