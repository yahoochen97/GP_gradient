% perform leave a few years out testing for baseline model

addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
addpath("~/Documents/Github/Bayesian-Causal-Inference/model");
startup;
clear all; close all;

for holdout_years=1:5
    % set random seed
    rng('default');

    load_PIRI_baseline_data;

    % init hyperparameter and define model
    PIRI_baseline_model;
    
    xs = x(x(:,1)>(m-holdout_years),:);
    ys = y(x(:,1)>(m-holdout_years));

    x = x(x(:,1)<=(m-holdout_years),:);
    y = y(x(:,1)<=(m-holdout_years));

    theta = minimize_v2(theta, @gp, p, inference_method, meanfunction, ...
                         covfunction, likfunction, x, y);

    [~, ~, fmu, fs2, lp] = gp(theta, inference_method, meanfunction, ...
                         covfunction, likfunction, x, y, xs, ys);
    dlmwrite("./results/baseline_holdout_" + int2str(holdout_years) + ".csv",lp,...
        'delimiter', ',', 'precision', 10);         
end                