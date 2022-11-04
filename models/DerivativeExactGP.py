import torch
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import LinearMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.models.exact_prediction_strategies import prediction_strategy
from gpytorch.means.mean import Mean
from gpytorch.kernels.kernel import Kernel
from gpytorch.lazy import InterpolatedLazyTensor
from gpytorch.utils.broadcasting import _mul_broadcast_shape

class ConstantVectorMean(Mean):
    def __init__(self, d=1, prior=None, batch_shape=torch.Size(), **kwargs):
        super(ConstantVectorMean, self).__init__()
        self.batch_shape = batch_shape
        self.register_parameter(name="constantvector", parameter=torch.nn.Parameter(torch.zeros(*batch_shape, d)))
        if prior is not None:
            self.register_prior("mean_prior", prior, "constantvector")

    def forward(self, input):
        # results = torch.zeros(input.shape)
        # for i in range(input.shape[0]):
        #     results[i] = self.constantvector[input[i].item()]
        return self.constantvector[input.int().reshape((-1,)).tolist()]

class myIndicatorKernel(Kernel):
    r"""
    A kernel for discrete indices. Kernel is defined by a lookup table.

    """

    def __init__(self, num_tasks, **kwargs):
        super().__init__(**kwargs)

        self.num_tasks = num_tasks

    def covar_matrix(self):
        res = torch.eye(self.num_tasks)
        return res

    def forward(self, i1, i2, **params):
        covar_matrix = self.covar_matrix()
        batch_shape = _mul_broadcast_shape(i1.shape[:-2], self.batch_shape)
        index1_shape = batch_shape + i1.shape[-2:]
        index2_shape = batch_shape + i2.shape[-2:]

        res = InterpolatedLazyTensor(
            base_lazy_tensor=covar_matrix,
            left_interp_indices=i1.long().expand(index1_shape),
            right_interp_indices=i2.long().expand(index2_shape),
        )
        return res


class ExactGPModel(gpytorch.models.ExactGP):
    """An exact Gaussian process (GP) model with a squared exponential 
    automataic releeance determination (SEArd) kernel.
    ExactGP: The base class of gpytorch for any Gaussian process latent function to be
        used in conjunction with exact inference.
    Attributes:
        train_x: (N x D) The training features X.
        train_i: (N x 1) The training unit index i.
        train_t: (N x 1) The training time index t.
        train_y: (N x 1) The training targets y.
        likelihood: gpytorch likelihood.
    """

    def __init__(
        self,
        n, T,
        train_x: torch.Tensor,
        train_i: torch.Tensor,
        train_t: torch.Tensor,
        train_y: torch.Tensor,
        likelihood=None
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        if likelihood is None:
            likelihood = GaussianLikelihood()
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPModel, self).__init__((train_x,train_i,train_t), train_y, likelihood)

        # self.mean_module = LinearMean(input_size=train_x[0].shape[1], bias=False)
        # self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x[0].shape[1]))
        self.i_mean_module =  ConstantVectorMean(d=n)
        self.t_mean_module = ConstantVectorMean(d=T)
        self.i_covar_module = ScaleKernel(myIndicatorKernel(n))
        self.t_covar_module = ScaleKernel(myIndicatorKernel(T))

        self.mean_module = LinearMean(input_size=train_x.shape[1])
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[1]))

    def forward(self, x, i , t):
        """Compute the prior latent distribution on a given input.
        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.
        Args:
            x: (n x D) The test points.
            i: (n x 1) The test unit index.
            t: (n x 1) The test time index.
        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x) + self.i_mean_module(i) + self.t_mean_module(t)
        covar_x = self.covar_module(x) + self.i_covar_module(i) + self.t_covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DerivativeExactGP(gpytorch.models.ExactGP):
    """Derivative of the ExactGPModel w.r.t. the test points x.
    Since differentiation is a linear operator this is again a Gaussian process.
    Attributes:
        train_x: (N x D) The training features X.
        train_i: (N x 1) The training unit index i.
        train_t: (N x 1) The training time index t.
        train_y: (N x 1) The training targets y.
        likelihood: gpytorch likelihood.
    """

    def __init__(
        self,
        train_x=None,
        train_i=None,
        train_t=None,
        train_y=None,
        likelihood=None
    ):
        """Inits GP model with data and a Gaussian likelihood."""
        if likelihood is None:
            likelihood = GaussianLikelihood()
        if train_y is not None:
            train_y = train_y.squeeze(-1)
        super(ExactGPModel, self).__init__((train_x,train_i,train_t), train_y, likelihood)

        self.D = train_x.shape[1]
        self.n = train_i.max().item()
        self.T = train_t.max().item()
        self.N = train_x.shape[0]
        self.train_inputs = (train_x, train_i, train_t)
        self.train_targets = train_y

        self.i_mean_module =  ConstantVectorMean(d=self.n)
        self.t_mean_module = ConstantVectorMean(d=self.T)
        self.i_covar_module = ScaleKernel(myIndicatorKernel(self.n))
        self.t_covar_module = ScaleKernel(myIndicatorKernel(self.T))

        self.mean_module = LinearMean(input_size=self.D)
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.D))

    def forward(self, x, i , t):
        """Compute the prior latent distribution on a given input.
        Typically, this will involve a mean and kernel function. The result must be a
        MultivariateNormal. Calling this model will return the posterior of the latent
        Gaussian process when conditioned on the training data. The output will be a
        MultivariateNormal.
        Args:
            x: (n x D) The test points.
            i: (n x 1) The test unit index.
            t: (n x 1) The test time index.
        Returns:
            A MultivariateNormal.
        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        mean_x = self.mean_module(x) + self.i_mean_module(i) + self.t_mean_module(t)
        covar_x = self.covar_module(x) + self.i_covar_module(i) + self.t_covar_module(t)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def get_L_lower(self):
        """Get Cholesky decomposition L, where L is a lower triangular matrix.
        Returns:
            Cholesky decomposition L.
        """
        return (
            self.prediction_strategy.lik_train_train_covar.root_decomposition()
            .root.evaluate()
            .detach()
        )

    def get_KXX_inv(self):
        """Get the inverse matrix of K(X,X).
        Returns:
            The inverse of K(X,X).
        """
        L_inv_upper = self.prediction_strategy.covar_cache.detach()
        return L_inv_upper @ L_inv_upper.transpose(0, 1)

    def get_KXX_inv_old(self):
        """Get the inverse matrix of K(X,X).
        Not as efficient as get_KXX_inv.
        Returns:
            The inverse of K(X,X).
        """
        X = self.train_inputs[0]
        sigma_n = self.likelihood.noise_covar.noise.detach()
        KXX = torch.eye(X.shape[0]) * sigma_n + self.covar_module(X).evaluate()
        return torch.inverse(KXX)

    def _get_KxX_dx(self, x):
        """Computes the analytic derivative of the kernel K(x,X) w.r.t. x.
        Args:
            x: (n x D) Test points.
        Returns:
            (n x D) The derivative of K(x,X) w.r.t. x.
        """
        X = self.train_inputs[0]
        n = x.shape[0]
        K_xX = self.covar_module(x, X).evaluate()
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        return (
            -torch.eye(self.D, device=x.device)
            / lengthscale ** 2
            @ (
                (x.view(n, 1, self.D) - X.view(1, self.N, self.D))
                * K_xX.view(n, self.N, 1)
            ).transpose(1, 2)
        )

    def _get_Kxx_dx2(self):
        """Computes the analytic second derivative of the kernel K(x,x) w.r.t. x.
        Args:
            x: (n x D) Test points.
        Returns:
            (n x D x D) The second derivative of K(x,x) w.r.t. x.
        """
        lengthscale = self.covar_module.base_kernel.lengthscale.detach()
        sigma_f = self.covar_module.outputscale.detach()
        return (
            torch.eye(self.D, device=lengthscale.device) / lengthscale ** 2
        ) * sigma_f

    def _get_mu_dx(self, x):
        """Get the derivative of mu(x).
        Returns:
            The gradient of mu(x) wrt x.
        """
        # x.require_grad = True
        # mu = self.mean_module(x)
        # for param in self.mean_module.parameters():
        #     param.requires_grad = False
        # # mu.requires_grad = True
        # mu.sum().backward()
        return self.mean_module.weights.expand(x.t().shape).t() # x.grad.data.reshape((-1,self.D))

    def posterior_derivative(self, x):
        """Computes the posterior of the derivative of the GP w.r.t. the given test
        points x.
        Args:
            x: (n x D) Test points.
        Returns:
            A GPyTorchPosterior.
        """
        mu_x_dx = self._get_mu_dx(x)

        with torch.no_grad():
            if self.prediction_strategy is None:
                train_output = super().__call__(self.train_inputs)
                # Create the prediction strategy for
                self.prediction_strategy = prediction_strategy(
                    train_inputs=self.train_inputs,
                    train_prior_dist=train_output,
                    train_labels=self.train_targets,
                    likelihood=self.likelihood,
                )

            K_xX_dx = self._get_KxX_dx(x)
            mu_prior = self.mean_module(self.train_inputs[0]) + self.i_mean_module(self.train_inputs[1]) + self.t_mean_module(self.train_inputs[2])
            mean_d = mu_x_dx + K_xX_dx @ self.get_KXX_inv() @ (self.train_targets - mu_prior)
            variance_d = (
                self._get_Kxx_dx2() - K_xX_dx @ self.get_KXX_inv() @ K_xX_dx.transpose(1, 2)
            )
            variance_d = variance_d.clamp_min(1e-8)

        return mean_d, variance_d