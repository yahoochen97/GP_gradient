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

% unit mean
meanfunction = {@meanMask, [0,1,0,0,0,0,0,0], {@meanDiscrete, n}};
theta.mean = zeros(n,1);
for i=1:n
    tmp = y(strcmp(data.country,countries{i}));
    theta.mean(i) = mean(tmp);
end

% unit trend
unit_length_scale = 4;
unit_output_scale = 0.5;
unit_error_covariance = {@covProd, {{@covMask, {1, {@covSEiso}}}, ...
                                    {@covMask, {2, {@covSEisoU}}}}};
theta.cov = [log(unit_length_scale); ... % 1
             log(unit_output_scale); ... % 2
             log(0.01)];                 % 3
         
% covariate effect (continuous)
% log(gdppc) and log(pop)
x_continuous_covariance = {@covMask, {[7,8], {@covSEard}}};
theta.cov = [theta.cov; ...
             log(std(log(data.gdppc)));...% 4
             log(std(log(data.pop))); ...% 5
             log(0.5)];                  % 6

% covariate effect (binary)
% cat_rat, ccpr_rat and democratic
x_binary_covariance = {@covMask, {[4,5,6], {@covSEard}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 7
             log(0.01); ...              % 8
             log(0.01); ...              % 9
             log(0.5)];                  % 10
         
% treatment effect (AIShame) 
AIShame_covariance = {@covMask, {[3], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 11
             log(0.5)];                  % 12


covfunction = {@covSum, {unit_error_covariance, x_continuous_covariance, ...
            x_binary_covariance, AIShame_covariance}};

% likfunction = {@likPoisson,'exp'};
likfunction = {@likGauss};
theta.lik = [log(0.05)];

prior.cov  = {[], ...
              [], ...
              [], ...
              [],...
              [],...
              [],...
              @priorDelta, ...
              @priorDelta, ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              []};  
prior.lik  = {[]};
prior.mean = cell(n,1);
prior.mean(:) = {@priorDelta};

inference_method = {@infPrior, @infExact, prior};
non_drift_idx = [2, 6, 10];
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
[post, ~, ~] = infExact(theta, meanfunction, covfunction, likfunction, x, y);

m_post = m_drift + K_drift*post.alpha;
tmp = K_drift.*post.sW;
K_post = K_drift - tmp'*solve_chol(post.L, tmp);

% AIShame effect
effect =  mean(m_post(data.AIShame==1)) - mean(m_post(data.AIShame==0));
effect_std =  sqrt(mean(diag(K_post)));

% unit trend prior
theta_drift = theta;
theta_drift.cov([6, 10, 12]) = log(0);
m_drift = feval(meanfunction{:}, theta_drift.mean, x);
K_drift = feval(covfunction{:}, theta_drift.cov, x);

m_post = m_drift + K_drift*post.alpha;
tmp = K_drift.*post.sW;
K_post = K_drift - tmp'*solve_chol(post.L, tmp);
K_post = diag(K_post);

unit_post_mu = zeros(n,m);
unit_post_std = zeros(n,m);
for i=1:n
    for j=1:m
       tmp = (strcmp(data.country,countries{i}) & data.year==years(j));
       if numel(m_post(tmp))~=0
           unit_post_mu(i,j) = m_post(tmp);
           unit_post_std(i,j) = sqrt(K_post(tmp));
       end
    end
end

% plot a few unit trends
[~,idx] = maxk(mean(treated,2),10);
for i = idx'
    fig = figure(i);
    clf;
    days = arrayfun(@(x) year_dict(x), data.year(strcmp(data.country,countries{i})));
    plot(days, unit_post_mu(i,days)); hold on;
    scatter(days, y(strcmp(data.country,countries{i})));
    title(countries{i});
    ylim([0,9]);
end
