## Simulation study of Gaussian Process regression framework for marginal effect estimation

### Data Generating process

Suppose we observe outcomes of $N=50$ respondents across $T=20$ different time periods with $D=2$ covariates. We generate outcome $y_{it}$ from static covariate effect and unit-specific time effect as follows:

$x_{id}\sim \mathcal{N}(0,1),\quad d=1,2$

$y_{it}=f(x_i)+g_i(t)+\varepsilon,\quad \varepsilon\sim\mathcal{N}(0,1)$

$f(x_i)=x^2_{i1}-x_{i1}*x_{i2}$

$g_i(t)\sim \mathcal{GP}(0,K_t),\quad i=1,\dots,N$

$K_t(t_1,t_2)=\exp(-\frac{1}{2}(t_1-t_2)^2)$

Above, the covariate effect $f(x)$ is static across time yet non-linear, and includes interactions. The time effect $g_i(t)$ is unit-specific and assumed to be generated from a Gaussian process.

### Model specification

We build a simple Gaussian process model that takes covariate $x$, time $t$ and unit-index $i$ as inputs with the following kernel structure:

$f(x,t,i)\sim\mathcal{GP}(0,K_x+K_t*K_i)$

$K_x(x,x')=\rho^2_x\exp(-\frac{1}{2\ell^2_1}(x_1-x'_1)^2-\frac{1}{2\ell^2_2}(x_2-x'_2)^2)$

$K_t(t_1,t_2)=\rho^2_t\exp(-\frac{1}{2\ell^2_t}(t_1-t_2)^2), \quad K_i(i,i')=1 \text{ if } i==i' \text{ else } 0$.

### Baselines


