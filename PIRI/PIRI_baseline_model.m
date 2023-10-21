% fix effect mean
year_meanfunction = {@meanMask, [1,0,0,0,0,0,0,0,0,0,0,0,0], {@meanDiscrete, m}};
theta.mean = [zeros(m,1)];

country_meanfunction = {@meanMask, [0,1,0,0,0,0,0,0,0,0,0,0,0], {@meanDiscrete, n}};
theta.mean = [theta.mean; zeros(n,1)];

meanfunction = {@meanSum, {year_meanfunction,country_meanfunction}};

% covariate effect (continuous)
% log(gdppc) and log(pop)
x_continuous_covariance = {@covMask, {[7,8], {@covSEard}}};
theta.cov = [log(std(data.gdppc));...    % 1
             log(std(data.pop)); ...     % 2
             log(0.5)];                  % 3

% covariate effect (binary)
% cat_rat, ccpr_rat, democratic, Civilwar2 and War
cat_binary_covariance = {@covMask, {[4], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 4
             log(0.5)];                  % 5
         
ccpr_binary_covariance = {@covMask, {[5], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 6
             log(0.5)];                  % 7
         
dem_binary_covariance = {@covMask, {[6], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 8
             log(0.5)];                  % 9

cw2_binary_covariance = {@covMask, {[9], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 10
             log(0.5)];                  % 11
         
war_binary_covariance = {@covMask, {[10], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 12
             log(0.5)];                  % 13
         
% treatment effect (AIShame) 
AIShame_covariance = {@covMask, {[3], {@covSEiso}}};
theta.cov = [theta.cov; ...
             log(0.01); ...              % 14
             log(0.5)];                  % 15

% country random effect
% country_random_covariance = {@covMask, {[2], {@covSEiso}}};
% theta.cov = [theta.cov; ... 
%              log(0.01); ...              % 16
%              log(0.5)];                  % 17

% PIRI effect
x_PIRI_covaraince = {@covMask, {[11], {@covSEiso}}};
theta.cov = [theta.cov; ... 
             log(std(data.PIRI)); ...    % 16
             log(0.5)];                  % 17
             
         
covfunction = {@covSum, {x_continuous_covariance, ...
            cat_binary_covariance, ccpr_binary_covariance,...
            dem_binary_covariance, ...
            cw2_binary_covariance, war_binary_covariance,...
            AIShame_covariance,...
            x_PIRI_covaraince}};

likfunction = {@likGauss};
theta.lik = [log(0.5)];

prior.cov  = {[], ...
              [], ...
              [], ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              [], ...
              @priorDelta, ...
              [], ...
              [], ...
              []};  
prior.lik  = {[]};
prior.mean = cell(m+n,1);
prior.mean(:) = {[]};

inference_method = {@infPrior, @infExact, prior};
non_drift_idx = [3,5,7, 9,11,13,17];
p.method = 'LBFGS';
p.length = 100;
