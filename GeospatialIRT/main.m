% Demonstration of dynamic Gaussian Process Item Response Theory
% with background characteristics and geospatial locations.

% loading GPML toolbox
clear;
close all;
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;

% background characteristics:
% x_1 ~ N(0,1), x_2 ~ N(0,1)
% normalized geospatial locations:
% g_1 ~ Unif(0,1), g_2 ~ Unif(0,1)
% geospatial correlated noise:
% ?(t) ~ N(0, K), K_{ij} = exp(-(g_i-g_j)^2/2), K_t = exp(-(t-t')^2/2) 
% latent policy position:
% ?_i(x,t,g) = x_1^2 + x_2 - x_1x_2 + cos(t) + ?_i(t)
% response model:
% y ~ binom(?_i,m), y=1,...,m

n = 100; % num of respondents
T = 10; % num of time periods
m = 10; % num of items in the battery

% generate data
rng('default');
x = normrnd(0,1,n,2);
g = unifrnd(0,1,n,2);
ts = (1:T)';
covfunc = {@covSEiso};             
hyp.cov = [log(0.5); log(1)];
% noise = mvnrnd(zeros(n*T,1),...
%     kron(feval(covfunc{:},hyp.cov,g),feval(covfunc{:},hyp.cov,ts./T/2)));
% theta = reshape(noise*0.1,[T,n])' + repmat(cos(ts'),n,1) + repmat(x(:,1).^2 + x(:,2) - x(:,1).*x(:,2),1,T);
noise = mvnrnd(zeros(n,1),feval(covfunc{:},hyp.cov,g))';
theta = repmat(noise,1,T) + repmat(cos(ts'),n,1) + repmat(x(:,1).^2 + x(:,2) - x(:,1).*x(:,2),1,T);

% normalize theta between 0 and 1
theta = (theta-min(theta,[],'all')) ./ (max(theta,[],'all')-min(theta,[],'all'));
y = zeros(n,T);
for i=1:n
    for t=1:T
        y(i,t) = binornd(m,theta(i,t));
    end
end

% build a hierachical GP model
train_x = zeros(n*T,2+2+1); % x,g,t
train_y = zeros(n*T,1); % y
for i=1:n
    for t=1:T
        train_x((i-1)*T+t,1:2) = x(i,:);
        train_x((i-1)*T+t,3:4) = g(i,:);
        train_x((i-1)*T+t,5) = t;
        train_y((i-1)*T+t) = y(i,t);
    end
end

meanfunc = {@meanZero}; 
x_covfunc = {@covMask,{[1,1,0,0,0], {@covSEard}}};
g_covfunc = {@covMask,{[0,0,1,1,0], {@covSEiso}}};
t_covfunc = {@covMask,{[0,0,0,0,1], {@covSEiso}}};
covfunc = {@covSum, {x_covfunc, g_covfunc, t_covfunc}};             
likfunc = {@likErf};
inffunc = {@infEP};
% hyp.mean = [-2;1;1];
hyp.mean = [0;0;0];
hyp.cov = [log(1);log(1)];
hyp.lik = [];

hyp = minimize(hyp, @gp, -10, @infEP, meanfunc, covfunc, likfunc, x, y);
