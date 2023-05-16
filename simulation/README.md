## Simulation study of Gaussian Process regression framework for marginal effect estimation

### Data Generating process

Suppose we observe outcomes of $N=50$ respondents across $T=20$ different time periods with $D=2$ covariates. We generate outcome $y_{it}$ from static covariate effect and time effect as follows:

$x_{id}\sim \mathcal{N}(0,1),\quad d=1,2$
$y_{it}=f(x_i)+g_i(t)+\varepsilon$
$f(x_i)=x^2_{i1}-x_{i1}*x_{i2}$
$g_i(t)\sim \mathcal{GP}(0,K_t),\quad i=1,\dots,N$
$K_t(t_1,t_2)=\exp(-\frac{1}{2}(t_1-t_2)^2)$
$\varepsilon\sim\mathcal{N}(0,1)$

Above, the covariate effect $f(x)$ is static across time yet non-linear, and includes interactions. The time effect $g_i(t)$ is unit-specific and assumed to be generated from a Gaussian process.

### Baselines


