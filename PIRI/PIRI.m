% PIRI gp model with unit trends
% PIRI ~ AIShame + u_i(t) + cat_rat + ccpr_rat 
%            + democratic + log(gdppc) + log(pop) 
%            + Civilwar2 + War
% u_i(t) ~ GP(b_i, K_t)


addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
addpath("~/Documents/Github/Bayesian-Causal-Inference/model");
startup;
clear all; close all;

% set random seed
rng('default');

load_PIRI_data;

% init hyperparameter and define model
PIRI_model;

theta = minimize_v2(theta, @gp, p, inference_method, meanfunction, ...
                     covfunction, likfunction, x, y);
    
% AIShame prior
theta_drift = theta;
theta_drift.cov(non_drift_idx) = log(0);
m_drift = feval(meanfunction{:}, theta_drift.mean, x)*0;
K_drift = feval(covfunction{:}, theta_drift.cov, x);

% AIShame posterior
[post, ~, ~] = infExact(theta, meanfunction, covfunction, likfunction, x, y);
[nlZ, ~] = gp(theta, inference_method, meanfunction, ...
                     covfunction, likfunction, x, y);

m_post = m_drift + K_drift*post.alpha;
tmp = K_drift.*post.sW;
K_post = K_drift - tmp'*solve_chol(post.L, tmp);

% AIShame effect
effect =  mean(m_post(data.AIShame==1)) - mean(m_post(data.AIShame==0));
effect_std =  sqrt(mean(diag(K_post)));

% unit trend prior
theta_drift = theta;
theta_drift.cov([18]) = log(0);
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

BIC = (2+3+6+1)*log(size(x,1)) + 2*nlZ;
fprintf("gp model\n");
fprintf("effect: %0.3f +- %0.3f\n", effect, effect_std);
fprintf("model evidence: %0.3f\n", -nlZ);
fprintf("BIC: %0.3f\n", BIC);

% plot a few unit trends
FONTSIZE = 16;
K = 17;
[~,idx] = maxk(mean(treated,2),K);
for i = idx'
    fig = figure(i);
    clf;
    days = arrayfun(@(x) year_dict(x), data.year(strcmp(data.country,countries{i})));
    plot(days, unit_post_mu(i,days)); hold on;
    scatter(days, y(strcmp(data.country,countries{i})));
    title(countries{i}, 'FontSize', FONTSIZE);
    ylim([0,9]);
    filename = "./results/" + countries{i} + ".pdf";
    print(fig, filename, '-dpdf','-r300');
    close;
end

% residual plot
[~, ~, fmu, ~] = gp(theta, inference_method, meanfunction, ...
                     covfunction, likfunction, x, y, x);

unit_post_mu = zeros(n,m);
for i=1:n
    for j=1:m
       tmp = (strcmp(data.country,countries{i}) & data.year==years(j));
       if numel(m_post(tmp))~=0
           unit_post_mu(i,j) = fmu(tmp);
       end
    end
end

fig = figure(1);
clf;
YTICKLABELS = cell(K,1);
DWs = [];
for i = 1:K
    days = arrayfun(@(x) year_dict(x), data.year(strcmp(data.country,countries{idx(i)})));
    rs = y(strcmp(data.country,countries{idx(i)})) - unit_post_mu(idx(i),days)';
    for j = 1:numel(days)
        plot([days(j);days(j)],[4*i;4*i+rs(j)],'b-','LineWidth', 2); hold on;
    end
    plot([1;18],[4*i,4*i], 'c-','LineWidth', 1);
    
    % Durbin-Watson Test
    % null hypothesis: residuals are uncorrelated. 
    [p,DW] = dwtest(rs,x(strcmp(data.country,countries{idx(i)})));
    DWs = [DWs, DW];
    fprintf("%s DW: %0.3f , p-value: %0.3f \n ", countries{idx(i)}, DW, p);
    
    % Ljung-Box Q-test
%     [h,p,~,~] = lbqtest(rs);
%     fprintf("%s LBQ: %0.3f , p-value: %0.3f \n ", countries{idx(i)}, h, p);

%     plot(days,4*i+rs,'b-','LineWidth', 1); hold on;
    YTICKLABELS{i} = countries{idx(i)};
%     title(countries{i}, 'FontSize', FONTSIZE);
%     ylim([0,9]);
end
xlim([0.8,numel(years)+0.2]);
yticks(4:4:(4*K));
yticklabels(YTICKLABELS);
h=gca; h.XAxis.TickLength = [0 0]; h.YAxis.TickLength = [0 0];
filename = "./results/gp_residuals.pdf";
set(fig, 'PaperPosition', [-1.5 -0.45 19 9]); 
set(fig, 'PaperSize', [16 8.5]);
print(fig, filename, '-dpdf','-r300');
close;
