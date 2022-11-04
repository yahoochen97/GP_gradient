import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.means import LinearMean
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
from models.DerivativeExactGP import DerivativeExactGP

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, ard_num_dim=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = LinearMean(input_size=1, bias=False)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dim=ard_num_dim))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def main():
    D = 2
    n = 100
    n_test = 50
    torch.random.manual_seed(12345)
    train_x = 8*torch.rand(n, D) - 4
    test_x1 = torch.cat((torch.linspace(-4, 4, n_test, requires_grad=True).reshape((-1,1)), torch.zeros(n_test,1)), 1)
    # True function is sin(x) + linear mean with Gaussian noise
    train_y = torch.sin(train_x[:,0]) + torch.cos(2*train_x[:,1]) + 2*train_x[:,0] - train_x[:,1] + torch.randn(n) * 0.1
    true_grad_1 = torch.cos(test_x1[:,0]) + 2
    test_x2 = torch.cat((torch.zeros(n_test,1), torch.linspace(-4, 4, n_test, requires_grad=True).reshape((-1,1))), 1)
    true_grad_2 = -2*torch.sin(2*test_x2[:,1]) - 1
    likelihood = GaussianLikelihood()
    model = DerivativeExactGP(D=D, likelihood=likelihood, ard_num_dims=2)
    model.append_train_data(train_x, train_y)
    
    # model = GPModel(train_x, train_y, likelihood)
    # hypers = {
    #     'likelihood.noise_covar.noise': torch.tensor(0.01),
    #     'covar_module.base_kernel.lengthscale': torch.tensor(1.),
    #     'covar_module.outputscale': torch.tensor(1.),
    #     'mean_module.weights': torch.tensor([1.,-1.0]),
    # }

    # model.initialize(**hypers)
    # train
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f  slope: %.3f   lengthscale: %.3f   outputscale: %.3f  noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.mean_module.weights[0].item(),
            model.covar_module.base_kernel.lengthscale[0,0].item(),
            model.covar_module.outputscale.item(),
            likelihood.noise.item()
        ))
        optimizer.step()

    print(model.covar_module.base_kernel.lengthscale)
    # evaluate
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x1))

    with torch.no_grad():
        mean_d1, variance_d1 = model.posterior_derivative(test_x1)
        variance_d1 = variance_d1.reshape((-1,1))

        mean_d2, variance_d2= model.posterior_derivative(test_x2)
        variance_d2 = variance_d2.reshape((-1,1))

    with torch.no_grad():
        # Initialize plot
        _, ax = plt.subplots(nrows=2,ncols=1,figsize=(6, 8))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        # plt.plot(train_x[:,0].numpy(), train_y.numpy(), 'k*', label='Data')
        # Plot predictive means as blue line
        # plt.plot(test_x[:,0].numpy(), observed_pred.mean.numpy(), 'b',label='Mean')

        # estimated marginal effect
        # lower = mean_d - 2*torch.sqrt(variance_d)
        # upper = mean_d + 2*torch.sqrt(variance_d)
        ax[0].plot(test_x1[:,0].numpy(), mean_d1[:,0].numpy(), '--k',label='Grad')
        ax[0].plot(test_x1[:,0].numpy(), true_grad_1.numpy(), '--r', label='True Grad')

        # Shade between the lower and upper confidence bounds
        # plt.fill_between(test_x[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence')

        # Shade between the lower and upper confidence bounds
        # ax.fill_between(test_x.numpy(), lower.reshape((-1,)).numpy(), upper.reshape((-1,)).numpy(), alpha=0.5)
        ax[0].legend()
        ax[0].set_ylim([-5, 5])
        ax[0].set_title('Estimated partial gradient of x1')

        ax[1].plot(test_x2[:,1].numpy(), mean_d2[:,1].numpy(), '--k',label='Grad')
        ax[1].plot(test_x2[:,1].numpy(), true_grad_2.numpy(), '--r', label='True Grad')
        ax[1].legend()
        ax[1].set_ylim([-5, 5])
        ax[1].set_title('Estimated partial gradient of x2')
        # ax[1].set_ylim([-6, 6])
        # ax[1].legend(['Gradient', 'Mean', 'Confidence'])
        # ax[1].set_title('Estimated gradient')
        plt.show()

if __name__ == "__main__":
    main()
