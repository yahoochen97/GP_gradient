addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
addpath("~/Documents/Github/Bayesian-Causal-Inference/model");
startup;
clear all; close all;

% set random seed
rng('default');

% read data
data = readtable("hb_data_complete.csv");

% clean data
% all zero PIRI for new zealand and netherland
data = data(~ismember(data.country, {'N-ZEAL','NETHERL'}),:);

countries = unique(data.country);
years = unique(data.year);

n = numel(countries);
m = numel(years);

treated = zeros(n,m);
for i=1:n
    for j=1:m
        if data.AIShame(data.year==years(j) & strcmp(data.country,countries(i))==1)
            treated(i,j)=1;
        end
    end
end

h = heatmap(treated, 'ylabel','country', 'ColorbarVisible', 'off');
cdl = h.YDisplayLabels;                                    % Current Display Labels
h.YDisplayLabels = repmat(' ',size(cdl,1), size(cdl,2));   % Blank Display Labels

country_dict = containers.Map(countries, 1:n);
year_dict = containers.Map(years, 1:m);

% x is:
% 1: year number
% 2: country id
% 3: AIShame (treatment indicator)
% 4: cat_rat
% 5: ccpr_rat
% 6: democratic
% 7: log(gdppc)
% 8: log(pop)
x = zeros(size(data,1), 8);
x(:,1) = arrayfun(@(x) year_dict(x), data.year);
x(:,2) = cellfun(@(x) country_dict(x), data.country);
% x(:,2) = 1;
% x(ismember(data.country, unique(data.country(data.PIRI==1))),2) = 2;
x(:,3) = data.AIShame;
x(:,4) = data.cat_rat;
x(:,5) = data.ccpr_rat;
x(:,6) = data.democratic;
x(:,7) = log(data.gdppc);
x(:,8) = log(data.pop);
y = data.PIRI;

% init hyperparameter and define model

% group trend
group_length_scale = 7;
group_output_scale = 0.5;
meanfunction = {@meanMask, [0,1,0,0,0,0,0,0], {@meanDiscrete, n}};
theta.mean = zeros(n,1);
for i=1:n
    tmp = y(strcmp(data.country,countries{i}));
    theta.mean(i) = mean(log(tmp(tmp~=0)));
end

% time covariance for group trends
time_covariance = {@covMask, {1, {@covSEiso}}};
theta.cov = [log(group_length_scale); ...      % 1
             log(group_output_scale)];         % 2

% nonlinear unit trend
unit_length_scale = 14;
unit_output_scale = 0.5;
unit_error_covariance = {@covProd, {{@covMask, {1, {@covSEiso}}}, ...
                                    {@covMask, {2, {@covSEisoU}}}}};
theta.cov = [theta.cov; ...
             log(unit_length_scale); ... % 3
             log(unit_output_scale); ... % 4
             log(0.01)];                 % 5
         
% marginalize unit mean
unit_mean_std = 0.5;
unit_mean_covariance = {@covMask, {2, {@covSEiso}}};

theta.cov = [theta.cov; ...
             log(0.01); ...              % 6
             log(unit_mean_std)];        % 7
         
% covariate effect (continuous)
% log(gdppc) and log(pop)
x_continuous_covariance = {@covMask, {[7,8], {@covSEard}}};
theta.cov = [theta.cov; ...
             log(std(log(data.gdppc)));...% 8
             log(std(log(data.pop))); ...% 9
             log(1)];                    % 10

% covariate effect (binary)
% cat_rat, ccpr_rat and democratic
x_binary_covariance = {@covMask, {[4,5,6], {@covSEard}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 11
             log(0.01); ...              % 12
             log(0.01); ...              % 13
             log(1)];                    % 14
         
% treatment effect (AIShame) 
AIShame_covariance = {@covMask, {[3], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 15
             log(1)];                    % 16


covfunction = {@covSum, {time_covariance, unit_mean_covariance, ...
            unit_error_covariance, x_continuous_covariance, ...
            x_binary_covariance, AIShame_covariance}};

likfunction = {@likPoisson,'exp'};
% likfunction = {@likGauss};
theta.lik = [];

prior.cov  = {{@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ... 
              [], ...
              {@priorTransform,@exp,@exp,@log,{@priorGamma,10,2}}, ...
              [], ...
              @priorDelta, ...
              @priorDelta, ...
              @priorDelta, ... 
              [],...
              [],...
              [],...
              @priorDelta, ...
              @priorDelta, ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              []};  
prior.lik  = {};
prior.mean = cell(n,1);
prior.mean(:) = {@priorDelta};

inference_method = {@infPrior, @infLaplace, prior};
non_drift_idx = [2, 4, 7, 10, 14];
p.method = 'LBFGS';
p.length = 100;

theta = minimize_v2(theta, @gp, p, inference_method, meanfunction, ...
                    covfunction, likfunction, x, y);
                
% AIShame prior
theta_drift = theta;
theta_drift.cov(non_drift_idx) = log(0);
m_drift = feval(meanfunction{:}, theta_drift.mean, x)*0;
K_drift = feval(covfunction{:}, theta_drift.cov, x);

% AIShame posterior
[post, ~, ~] = infLaplace(theta, meanfunction, covfunction, likfunction, x, y);
% [post, ~, ~] = infExact(theta, meanfunction, covfunction, likfunction, x, y);

m_post = m_drift + K_drift*post.alpha;
tmp = K_drift.*post.sW;
K_post = K_drift - tmp'*solve_chol(post.L, tmp);

% AIShame effect
effect =  mean(m_post(data.AIShame==1)) - mean(m_post(data.AIShame==0));
effect_std =  sqrt(mean(diag(K_post)));