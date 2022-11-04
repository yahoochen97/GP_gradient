% Gaussian process regression with gradient estimation

% load gpml
% add gpml path
gpml_path = "/Users/yahoo/Documents/WashU/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07";
addpath(gpml_path);
startup;
rng("default");

% prepare the plotting environment
clear; close all;
write_fig = 1;

% Set the sample size.
n = 50;

% Likelihood function for the sample and priors for hyperparameters
likfunc = {@likGauss}; 
sn = 0.1;
hyp.lik = log(sn);

% Generate n inputs for a data generating process
x = unifrnd(-2,2,n,1);

% Covariance function and priors for hyperparameters
covfunc = {@covSEiso};
ell = 1/2;
sf = 1;
% The isotropic squared exponential function has two hyperparamters.
hyp.cov = log([ell; sf]);

% Mean function and priors for hyperparameters
meanfunc = {@meanSum, {@meanLinear, @meanConst}};
hyp.mean = [zeros(size(x,2), 1), 0];
% Generate n outputs of a data generating process
K = feval(covfunc{:}, hyp.cov, x) + 0.001*eye(n);
mu = feval(meanfunc{:}, hyp.mean, x);
% y = chol(K)'*normrnd(0,1,n,1) + mu + exp(hyp.lik)*normrnd(0,1,n,1);
y = sin(pi*x) + mu + exp(hyp.lik)*normrnd(0,1,n,1);

% Make a grid in the x dimension on which to predict values
z = linspace(-5, 5, 101)';

% Define hyperpriors
prior.cov = {{@priorTransform,@exp,@exp,@log,{@priorGamma,1,1}},... % Gamma prior on ell
               {@priorTransform,@exp,@exp,@log,{@priorGamma,1,2}}}; % Gamma prior on sf
prior.mean = { {@priorGauss ,1,1}, ... % Gaussian prior on mean slope
                {@priorGauss ,0,1}}; % Gaussian prior on mean constant
prior.lik = {{@priorTransform,@exp,@exp,@log,{@priorGamma,0.01,10}}}; % Gamma prior on noise

inffunc = {@infPrior, @infExact, prior};
% no hyperprior
inffunc = {@infExact};
p.method = 'LBFGS';
p.length = 100;

% Find the maximum a posteriori estimates for the model hyperparameters.
hyp = minimize_v2(hyp, @gp, p, inffunc, meanfunc, covfunc, likfunc, x, y);
fprintf("MAP ell: %.3f\n", exp(hyp.cov(1)));
fprintf("MAP sf: %.3f\n", exp(hyp.cov(2)));
fprintf("MAP slope: %.3f\n", (hyp.mean(1)));
fprintf("MAP constant: %.3f\n", (hyp.mean(2)));
fprintf("MAP noise: %.3f\n", exp(hyp.lik));

% Report negative log marginal likelihood of the data with the optimized
% model hyperparameters.
[m, s2] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, z);

[post, ~, ~] = feval(inffunc{:}, hyp, meanfunc, covfunc, likfunc, x, y);
% squared exponential kernel first gradient
dK = feval(covfunc{:},hyp.cov,x,z)/exp(2*hyp.cov(1));
dK = dK.*repmat(x,1,numel(z)) - dK.*repmat(z',numel(x),1);

% squared exponential kernel second gradient
d2K = feval(covfunc{:},hyp.cov,z,z)/exp(2*hyp.cov(1)).*(ones(numel(z),numel(z))-sq_dist(z,z)/exp(2*hyp.cov(1)));

% gradient gp mean
dm = hyp.mean(1) + dK'*post.alpha;

% gradient gp cov
tmp = post.L'\(repmat(post.sW,1,numel(z)).*dK);
ds2 = d2K - tmp'*tmp;

% Plot predicted values and credible intervals
fig = figure(1);
% tiledlayout(1,2,'Padding', 'none', 'TileSpacing', 'compact');
% nexttile;
set(gca, 'FontSize', 24);
f = [m+2*sqrt(s2); flipdim(m-2*sqrt(s2),1)];
fill([z; flipdim(z,1)], f, [64, 214, 247] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none');
hold on; plot(z, m, 'LineWidth', 2); plot(x, y, '+', 'MarkerSize', 12);
grid on;
plot(z,dm, '--', 'LineWidth', 2);
f = [dm+2*sqrt(diag(ds2)); flipdim(dm-2*sqrt(diag(ds2)),1)];
fill([z; flipdim(z,1)], f, [6,6,6]/8, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none');
xlabel('Input (x)');
ylabel('Output (y)');
title('y=sin({\pi}x)+{\epsilon}');
legend('function 2{\sigma} CI',...
     'posterior mean','training data','posterior gradient', 'gradient 2{\sigma} CI',...
     'Location', 'northwest', 'NumColumns', 1);
legend boxoff;

if write_fig
    set(fig, 'PaperPosition', [0 0 10 8]); 
    set(fig, 'PaperSize', [10 8]); 

    filename = "./gradient.pdf";
    print(fig, filename, '-dpdf','-r300', '-fillpage');
    close;
end

covfunc = {@covSum, {@covSEiso, @covConst, @covLINiso, {@covPoly,2}}}; % 
ell = 1;
sf = 1;
% The isotropic squared exponential function has two hyperparamters.
hyp.cov = [log(ell); log(sf); log(1); log(1); log(0); log(0)];

beta = mvnrnd(zeros(2,1),eye(2),1)';
y = [ones(n,1),x]*beta + 0.1*normrnd(0,1,n,1);
x_ = [ones(n,1),x,x.^2];
z_ = [ones(101,1),z,z.^2];
K_ = feval(covfunc{:},hyp.cov,z,x);
mu = K_*inv(0.01*eye(50)+feval(covfunc{:},hyp.cov,x))*y;
K = feval(covfunc{:},hyp.cov,z) - K_*inv(0.01*eye(50)+feval(covfunc{:},hyp.cov,x))*K_' ;
% f = mvnrnd(mu,K + 0.01*eye(101));
scatter(z,mu);

