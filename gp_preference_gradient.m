clear;
close all;
addpath("~/Documents/Washu/CSE515T/Code/Gaussian Process/gpml-matlab-v3.6-2015-07-07");
startup;
addpath("utilities");
FONTSIZE=16;

rng('default');

% define function
n = 101;
x = linspace(-2, 2, n)';                 
f = x.^2+x-2;
p = normcdf(f);
y_grad = normpdf(f).*(2*x+1);
y = 2*binornd(1,p)-1;

% visualize data
fig=figure(1); tiledlayout(2,2);
nexttile;
plot(x,p); hold on; scatter(x(y==1),y(y==1), 'r*');
scatter(x(y==-1),y(y==-1)+1, 'k*');
legend(["p(y=1|x)", "y=1","y=0"],...
    'Location', 'northwest','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');
title("observations", 'FontSize', FONTSIZE);

% build a gp model
meanfunc = {@meanSum, {@meanConst, {@meanPoly, 2}}};   
covfunc = {@covSEiso};             
likfunc = {@likErf};
inffunc = {@infEP};
% hyp.mean = [-2;1;1];
hyp.mean = [0;0;0];
hyp.cov = [log(1);log(1)];
hyp.lik = [];

prior.cov  = {@priorDelta, @priorDelta,};  
prior.lik  = {};
prior.mean = {[],[],[]};

% hyp = minimize(hyp, @gp, -10, @infEP, meanfunc, covfunc, likfunc, x, y);
[~,~,fmu,fs2] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, x);

% visualize posterior gp
nexttile;
plot(x,p); hold on; plot(x,normcdf(fmu./sqrt(1+fs2)), '--');
f = [normcdf(fmu+2*sqrt(fs2)); flipdim(normcdf(fmu-2*sqrt(fs2)),1)];
fill([x; flipdim(x,1)], f, [6,6,6]/8, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none');
legend(["true p(y=1|x)", "posterior mean","posterior 95% CI"],...
    'Location', 'northwest','NumColumns',1, 'FontSize',FONTSIZE);
legend('boxoff');
title("posterior of p(y=1|x)", 'FontSize', FONTSIZE);

% sample from gp posterior
nexttile;
N = 1001;
xs = linspace(-2,2,N)';
plot(x, y_grad, '-', 'LineWidth',2); hold on;

[~,~,fmu,fs2, ~, post] = gp(hyp, @infEP, meanfunc, covfunc, likfunc, x, y, xs);

dK = feval(covfunc{:},hyp.cov,x,xs)/exp(2*hyp.cov(1));
dK = dK.*repmat(x,1,numel(xs)) - dK.*repmat(xs',numel(x),1);

% squared exponential kernel second gradient
d2K = feval(covfunc{:},hyp.cov,xs,xs)/exp(2*hyp.cov(1)).*(ones(numel(xs),numel(xs))...
    -sq_dist(xs,xs)/exp(2*hyp.cov(1)));

% gradient gp posterior mean
dm = dK'*post.alpha; % + hyp.mean(2) + 2*hyp.mean(3)*xs;
% dm = dm.*normpdf(fmu);

% gradient gp posterior cov
tmp = post.L'\(repmat(post.sW,1,numel(xs)).*dK);
ds2 = d2K - tmp'*tmp;
% ds2 = ds2.*(normpdf(fmu)*normpdf(fmu)');

% Gauss-Hermite quadrature
n_gauss_hermite = 5;
[ks,ws] = root_GH(n_gauss_hermite);

% dK_D = -2*(dK'/post.L)*tmp;
% L = chol(K+1e-12*eye(N));
% tmp = L'\dK_D;

for i=1:n_gauss_hermite
    f_bar = sqrt(2)*ks(i)*sqrt(fs2) + fmu;
    mu_bar = normpdf(f_bar).*dm;
    K_bar = normpdf(f_bar).^2.*diag(ds2); 
    % plot dp(y=1)/dx as phi(f)*df/dx
    % plot(xs,normpdf(f_bar).*dm, '--', 'LineWidth', 2);
    f = [(mu_bar+2*sqrt(K_bar));...
        flipdim(mu_bar-2*sqrt(K_bar),1)];
    fill([xs; flipdim(xs,1)], f, [5,5,5]/8, ...
         'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
         'edgecolor', 'none');
end
ylim([-3,2]);
labels = ["true gradient"];
for i=1:n_gauss_hermite
   labels = [labels,sprintf("weight: %0.3f",ws(i))];
end
legend(labels,...
    'Location', 'southeast','NumColumns',3, 'FontSize',FONTSIZE);
legend('boxoff');
title("posterior of dp(y=1)/dx", 'FontSize', FONTSIZE);

% empirical sample f and df/dx
nexttile;
% f = [dm+2*sqrt(diag(ds2)); flipdim(dm-2*sqrt(diag(ds2)),1)];
% fill([xs; flipdim(xs,1)], f, [6,6,6]/8, ...
%      'facealpha', 0.5, ...
%      'edgecolor', 'none'); hold on;
 
tmp = post.L'\(repmat(post.sW,1,numel(xs)).*feval(covfunc{:}, hyp.cov,x,xs));
K = feval(covfunc{:}, hyp.cov, xs) - tmp'*tmp;
gs = zeros(9,N);
dists = zeros(9,1);
for it=1:9
    f = mvnrnd(fmu, K);
    f_grad = (f(2:N) - f(1:(N-1)))*(N-1)/4; f_grad(N) = f_grad(end);
    g = normpdf(f).*f_grad;
    dist = mean((g'-normpdf(xs.^2+xs-2).*(2*xs+1)).^2);
%     plot(xs, g); hold on;
    gs(it,:) = g;
    dists(it) = dist;
end 

[dists, idx]=sort(dists);
gs = gs(idx,:);
labels=[];
for it=1:9
    labels = [labels,sprintf("dist: %0.3f",dists(it))];
    c = [2,2,2]/8+dists(it)/max(dists)*([5,5,5]/8);
    plot(xs, gs(it,:), 'color', c); hold on;
end 

ylim([-3,2]);

legend(labels,...
    'Location', 'southeast','NumColumns',3, 'FontSize',FONTSIZE);
legend('boxoff');

title("samples from dp(y=1)/dx", 'FontSize', FONTSIZE);
filename = "./results/GMM_example.pdf";
set(fig, 'PaperPosition', [-1 -0.45 19 6]); 
set(fig, 'PaperSize', [17 5.5]);
print(fig, filename, '-dpdf','-r300');
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% compare mutual information I(y,g;x,D) with I(y,p;x,D)
% define function 
hyp.mean = [-1;0;1];
hyp.cov = [log(1);log(2)];
% x = [-2;-1.0;-0.5;0.0; 1.0; 1.5;2];
% y = [1;-1;-1;-1;-1;1;1];
n = 6;
x = [-2;min(max(normrnd(-1.0,0.25,n,1),-2),2);...
    min(max(normrnd(0,0.25,n,1),-2),2);...
    min(max(normrnd(1.5,0.1,n,1),-2),2);2];
f = 4*x.^2-2;
p = normcdf(f);
y = 2*binornd(1,p)-1;
% x = [-2;x;2];
% y = [1;y;-1];

N = 101;
xs = linspace(-2,2,N)';

% visualize data
fig=figure(2); tiledlayout(2,2);
nexttile;

% clear p;
% p.length = 100;
% p.method = 'LBFGS';
% p.verbosity = 0;
% inference_method = {@infPrior, @infEP, prior};
% hyp = minimize(hyp, @gp, -5, inference_method, meanfunc, covfunc, likfunc, x, y);

[~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
% plot(xs,fmu-2*sqrt(fs2)); hold on; plot(xs,fmu+2*sqrt(fs2)); plot(xs,sqrt(fs2))

% visualize posterior gp
plot(xs, normcdf(fmu./sqrt(1+fs2)), 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;
f = [normcdf(fmu+2*sqrt(fs2)); flipdim(normcdf(fmu-2*sqrt(fs2)),1)];
% f = [(ymu+1)/2+2*sqrt(ys2); (ymu+1)/2-2*sqrt(ys2)];
fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none');
scatter(x(y==1),y(y==1), 'r*');
scatter(x(y==-1),y(y==-1)+1, 'k*');
legend(["posterior mean","posterior 95% CI","y=1","y=0"],... 
    'Location', 'southwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("p(y=1|x)", 'FontSize',FONTSIZE);

% plot dp(y=1)/dx posterior
nexttile;
% current GMM
[ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,inffunc,meanfunc, likfunc, x, y, xs); 

mu_bar = zeros(N,1);
g_s2 = zeros(N,1);
for i=1:n_gauss_hermite
    mu_bar = mu_bar + ws(i)*mu_GMM{i};
    g_s2 = g_s2 + (ws(i)*sigma_GMM{i}).^2;
end
g_s2 = sqrt(g_s2);

plot(xs,mu_bar, 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;

for i=1:n_gauss_hermite
    mu_bar = mu_GMM{i};
    f = [(mu_bar+2*sigma_GMM{i});...
        flipdim(mu_bar-2*sigma_GMM{i},1)];
    fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
         'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
         'edgecolor', 'none');
end
ylim([-2.5,2.5]);
labels = ["posterior mean"];
for i=1:n_gauss_hermite
   labels = [labels,sprintf("weight: %0.3f",ws(i))];
end
legend(labels,...
    'Location', 'northwest','NumColumns',3, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("dp(y=1)/dx", 'FontSize',FONTSIZE);

% mutual information I(y,p;x,D)
nexttile;
ps = normcdf(fmu./sqrt(1+fs2));
C = sqrt(pi*log(2)/2);
IG_p = -ps.*log2(ps) - (1-ps).*log2(1-ps) - ...
    C./sqrt(C^2+fs2).*exp(-fmu.^2./(C^2+fs2)/2);
f = [IG_p; xs*0];

plot(xs, IG_p, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx] = max(IG_p);
scatter(xs(idx),IG_p(idx)+0.01,25,'v','MarkerFaceColor',[0 0 0]);
ylim([0,0.3]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["I(y,f)", "next location x = " + xs(idx)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel("BALD", 'FontSize',FONTSIZE);

% mutual information I(y,g;x,D)
nexttile;

% H[y | x, D] part is the same as BALD
IG_g = -ps.*log2(ps) - (1-ps).*log2(1-ps);
% IG_g = IG_g*0;

%  compute E_g[H[y | x, g, D]]
n_gauss_hermite = 30;
[ks,~] = root_GH(n_gauss_hermite);

% current GMM
[ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,inffunc,meanfunc, likfunc, x, y, xs); 

for k=1:N
    % compute new GMMs
    [~,mu_GMM1,sigma_GMM1] = grad_GMM(n_gauss_hermite,hyp,...
                inffunc,meanfunc, likfunc, [x;xs(k)], [y;1], xs(k)); 
    [~,mu_GMM0,sigma_GMM0] = grad_GMM(n_gauss_hermite,hyp,...
        inffunc,meanfunc, likfunc, [x;xs(k)], [y;-1], xs(k)); 
    for i=1:n_gauss_hermite
        for j=1:n_gauss_hermite
            g_bar = sqrt(2)*sigma_GMM{i}(k)*ks(j) + mu_GMM{i}(k);
            p_1 = 0; p_0 = 0;
            % compute p(g|y,x,D)
            for it=1:n_gauss_hermite
                p_1 = p_1 + ws(it)*normpdf(g_bar,mu_GMM1{it},sigma_GMM1{it});
                p_0 = p_0 + ws(it)*normpdf(g_bar,mu_GMM0{it},sigma_GMM0{it});
            end 
            % compute E[H[y|x,g]]  
            p_k = p_1*ps(k)/(p_1*ps(k)+p_0*(1-ps(k)));
            p_k = max(min(p_k,1-1e-12),1e-12);
            h = -p_k*log2(p_k) - (1-p_k)*log2(1-p_k);
            if ~isnan(h), IG_g(k) = IG_g(k) - ws(i)*ws(j)*h; end
        end
    end   
end
% plot(xs,-IG_g); hold on; plot(xs, -ps.*log2(ps) - (1-ps).*log2(1-ps));
f = [IG_g; xs*0];

plot(xs, IG_g, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx_g] = max(IG_g);
scatter(xs(idx_g),IG_g(idx_g)+0.01,25,'v','MarkerFaceColor',[0 0 0]);
ylim([0,0.15]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["I(y,g)", "next location x = " + xs(idx_g)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');

ylabel("GRADBALD", 'FontSize',FONTSIZE);

filename = "./results/MI_example.pdf";
set(fig, 'PaperPosition', [-0.7 -0.3 25 6]); 
set(fig, 'PaperSize', [23 5.4]);
print(fig, filename, '-dpdf','-r300');
close;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% acquisition UCB, DE-U, BALD
hyp.mean = [-1;0;1];
hyp.cov = [log(1);log(2)];
n = 6;
x = [-2;min(max(normrnd(-1.0,0.25,n,1),-2),2);...
    min(max(normrnd(0,0.25,n,1),-2),2);...
    min(max(normrnd(1.5,0.1,n,1),-2),2);2];
f = 4*x.^2-2;
p = normcdf(f);
y = 2*binornd(1,p)-1;

N = 101;
xs = linspace(-2,2,N)';
n_gauss_hermite = 5;

% visualize data
fig=figure(2); tiledlayout(1,2);

[~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);

% plot dp(y=1)/dx posterior
% current GMM
[ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,inffunc,meanfunc, likfunc, x, y, xs); 

mu_bar = zeros(N,1);
g_s2 = zeros(N,1);
for i=1:n_gauss_hermite
    mu_bar = mu_bar + ws(i)*mu_GMM{i};
    g_s2 = g_s2 + (ws(i)*sigma_GMM{i}).^2;
end
g_s2 = sqrt(g_s2);

% DE-ME
nexttile;
IG_p = sqrt(g_s2);
f = [IG_p; xs*0];

plot(xs, IG_p, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx] = max(IG_p);
scatter(xs(idx),IG_p(idx)+0.01,25,'v','MarkerFaceColor',[0 0 0]);
ylim([0,0.6]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["DE-ME", "next location x = " + xs(idx)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel('{\alpha}(x,x^{(j)};D)', 'FontSize',FONTSIZE);

% mutual information I(y,p;x,D): BALD
nexttile;
ps = normcdf(fmu./sqrt(1+fs2));
C = sqrt(pi*log(2)/2);
IG_p = -ps.*log2(ps) - (1-ps).*log2(1-ps) - ...
    C./sqrt(C^2+fs2).*exp(-fmu.^2./(C^2+fs2)/2);
f = [IG_p; xs*0];

plot(xs, IG_p, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx] = max(IG_p);
scatter(xs(idx),IG_p(idx)+0.01,25,'v','MarkerFaceColor',[0 0 0]);
ylim([0,0.3]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["BALD", "next location x = " + xs(idx)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel('{\alpha}(x,x^{(j)};D)', 'FontSize',FONTSIZE);


filename = "./results/acquisition_example.pdf";
set(fig, 'PaperPosition', [-0.7 -0.1 25 3]); 
set(fig, 'PaperSize', [23 2.8]);
print(fig, filename, '-dpdf','-r300');
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fig = figure(3); tiledlayout(3,2);
n_gauss_hermite=5;
for idx_=[idx,idx_g] 
    nexttile; 
    % current GMM
    [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,...
        inffunc,meanfunc, likfunc, x, y, xs); 
    
    mu_bar = zeros(N,1);
    for i=1:n_gauss_hermite
        mu_bar = mu_bar + ws(i)*mu_GMM{i};
    end

    plot(xs,mu_bar, 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;

    for i=1:n_gauss_hermite
        mu_bar = mu_GMM{i};
        f = [(mu_bar+2*sigma_GMM{i});...
            flipdim(mu_bar-2*sigma_GMM{i},1)];
        fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
             'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
             'edgecolor', 'none');
    end
    ylim([-2.5,2.5]);
    xline(xs(idx_),'--r');
    b = gca; legend(b,'off');
    ylabel("dp(y=1)/dx", 'FontSize',FONTSIZE);
    if idx_==idx
        title("BALD x = "+ xs(idx_));
    else
        title("GRADBALD x = "+ xs(idx_));
    end
end

for y_=[0,1]   
    for idx_=[idx,idx_g]  
        nexttile; 
        % current GMM
        [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,...
            inffunc,meanfunc, likfunc, [x;xs(idx_)], [y;2*y_-1], xs); 
         mu_bar = zeros(N,1);
        for i=1:n_gauss_hermite
            mu_bar = mu_bar + ws(i)*mu_GMM{i};
        end

        plot(xs,mu_bar, 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;
        
        for i=1:n_gauss_hermite
            mu_bar = mu_GMM{i};
            f = [(mu_bar+2*sigma_GMM{i});...
                flipdim(mu_bar-2*sigma_GMM{i},1)];
            fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
                 'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
                 'edgecolor', 'none');
        end
        ylim([-2.5,2.5]);
        xline(xs(idx_),'--r');
        b = gca; legend(b,'off');
        ylabel("dp(y=1)/dx", 'FontSize',FONTSIZE);
        title("observe y = "+y_ + " at x = " + xs(idx_)...
            + " with prob " + (round(ps(idx_),3)*y_+(1-round(ps(idx_),3))*(1-y_)));
    end
end

filename = "./results/MI_g_example.pdf";
set(fig, 'PaperPosition', [-0.7 -0.4 25 9]); 
set(fig, 'PaperSize', [23 8.4]);
print(fig, filename, '-dpdf','-r300');
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure(4); tiledlayout(3,2);
n_gauss_hermite=30;
for idx_=[idx,idx_g] 
    nexttile; 
    % current GMM
    [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,...
        inffunc,meanfunc, likfunc, x, y, xs(idx_)); 
    
    gmm_pdf = zeros(N,1);
    
    for i=1:n_gauss_hermite
        if sigma_GMM{i}<=1e-3, continue; end
        gmm_pdf = gmm_pdf + ws(i)*normpdf(xs,mu_GMM{i},sigma_GMM{i});
        f = [ws(i)*normpdf(xs,mu_GMM{i},sigma_GMM{i});...
            zeros(N,1)];
        fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
             'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
             'edgecolor', 'none'); hold on;
    end
    plot(xs,gmm_pdf, 'LineWidth',2, 'color', [40, 150, 215] / 255);
%     ylim([0,2]);
    b = gca; legend(b,'off');
    ylabel("dp(y=1)/dx pdf", 'FontSize',FONTSIZE);
    if idx_==idx
        title("BALD x = "+ xs(idx_));
    else
        title("GRADBALD x = "+ xs(idx_));
    end
end

for y_=[0,1]   
    for idx_=[idx,idx_g]  
        nexttile; 
        % current GMM
        [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,...
            inffunc,meanfunc, likfunc, [x;xs(idx_)], [y;2*y_-1], xs(idx_)); 
        
         gmm_pdf = zeros(N,1);
    
        for i=1:n_gauss_hermite
            if sigma_GMM{i}<=1e-3, continue; end
            gmm_pdf = gmm_pdf + ws(i)*normpdf(xs,mu_GMM{i},sigma_GMM{i});
            f = [ws(i)*normpdf(xs,mu_GMM{i},sigma_GMM{i});...
                zeros(N,1)];
            fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
                 'facealpha', 0.4*ws(i)/max(ws)+0.1, ...
                 'edgecolor', 'none'); hold on;
        end
        plot(xs,gmm_pdf, 'LineWidth',2, 'color', [40, 150, 215] / 255);
%         ylim([0,2.0]);
        b = gca; legend(b,'off');
        ylabel("dp(y=1)/dx pdf", 'FontSize',FONTSIZE);
        title("observe y = "+y_ + " at x = " + xs(idx_) ...
            + " with prob " + (round(ps(idx_),3)*y_+(1-round(ps(idx_),3))*(1-y_)));
    end
end

filename = "./results/MI_gmm_example.pdf";
set(fig, 'PaperPosition', [-0.7 -0.4 25 6]); 
set(fig, 'PaperSize', [23 5.4]);
print(fig, filename, '-dpdf','-r300');
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure(5); tiledlayout(3,2);
for idx_=[idx,idx_g] 
    nexttile; 
    [~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);

    % visualize posterior gp
    plot(xs, normcdf(fmu./sqrt(1+fs2)), 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;
    f = [normcdf(fmu+2*sqrt(fs2)); flipdim(normcdf(fmu-2*sqrt(fs2)),1)];
    % f = [(ymu+1)/2+2*sqrt(ys2); (ymu+1)/2-2*sqrt(ys2)];
    fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
         'facealpha', 0.5, ...
         'edgecolor', 'none');
    scatter(x(y==1),y(y==1), 'r*');
    scatter(x(y==-1),y(y==-1)+1, 'k*');
    xline(xs(idx_),'--r');
%     legend(["posterior mean","posterior 95% CI","y=1","y=0",''],... 
%         'Location', 'northwest','NumColumns',4, 'FontSize',FONTSIZE);
%     legend('boxoff');
b = gca; legend(b,'off');
    ylabel("p(y=1|x)", 'FontSize',FONTSIZE);
     if idx_==idx
        title("BALD x = "+ xs(idx_));
    else
        title("GRADBALD x = "+ xs(idx_));
    end
end

for y_=[0,1]   
    for idx_=[idx,idx_g]  
        x_new = [x;xs(idx_)];
        y_new = [y;2*y_-1];
        nexttile; 
        [~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc,...
            covfunc, likfunc, x_new, y_new , xs);

        % visualize posterior gp
        plot(xs, normcdf(fmu./sqrt(1+fs2)), 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;
        f = [normcdf(fmu+2*sqrt(fs2)); flipdim(normcdf(fmu-2*sqrt(fs2)),1)];
        % f = [(ymu+1)/2+2*sqrt(ys2); (ymu+1)/2-2*sqrt(ys2)];
        fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
             'facealpha', 0.5, ...
             'edgecolor', 'none');
        scatter(x_new(y_new==1),y_new(y_new==1), 'r*');
        scatter(x_new(y_new==-1),y_new(y_new==-1)+1, 'k*');
        xline(xs(idx_),'--r');
%         legend(["posterior mean","posterior 95% CI","y=1","y=0",''],... 
%             'Location', 'northwest','NumColumns',4, 'FontSize',FONTSIZE);
%         legend('boxoff');
        b = gca; legend(b,'off');
        ylabel("p(y=1|x)", 'FontSize',FONTSIZE);
       title("observe y = "+y_ + " at x = " + xs(idx_)...
        + " with prob " + (round(ps(idx_),3)*y_+(1-round(ps(idx_),3))*(1-y_)));
    end
end

filename = "./results/MI_y_example.pdf";
set(fig, 'PaperPosition', [-0.7 -0.4 25 9]); 
set(fig, 'PaperSize', [23 8.4]);
print(fig, filename, '-dpdf','-r300');
close;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fig = figure(5); tiledlayout(5,5);
% IG = 0;
% n_gauss_hermite = 20;
% [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,...
%             inffunc,meanfunc, likfunc, x,y, xs(idx_g)); 
% for i=1:n_gauss_hermite
%    for j=1:n_gauss_hermite
% %        nexttile;
%        g_bar = sqrt(2)*sigma_GMM{i}*ks(j) + mu_GMM{i};
%        [~,mu_GMM1,sigma_GMM1] = grad_GMM(n_gauss_hermite,hyp,...
%                 inffunc,meanfunc, likfunc, [x;xs(idx_g)], [y;1], xs(idx_g)); 
%         [~,mu_GMM0,sigma_GMM0] = grad_GMM(n_gauss_hermite,hyp,...
%             inffunc,meanfunc, likfunc, [x;xs(idx_g)], [y;-1], xs(idx_g)); 
%         p_1 = 0; p_0 = 0;
%         % compute p(g|y,x,D)
%         for it=1:n_gauss_hermite
%             p_1 = p_1 + ws(it)*normpdf(g_bar,mu_GMM1{it},sigma_GMM1{it});
%             p_0 = p_0 + ws(it)*normpdf(g_bar,mu_GMM0{it},sigma_GMM0{it});
%         end 
%         % compute E[H[y|x,g]]  
%         p_k = p_1*ps(idx_g)/(p_1*ps(idx_g)+p_0*(1-ps(idx_g)));
%         hs = -ps(idx_g)*log2(ps(idx_g))-(1-ps(idx_g))*log2(1-ps(idx_g));
%         h = -p_k*log2(p_k)-(1-p_k)*log2(1-p_k);
% %         scatter(0,ps(idx_g),'o'); hold on; scatter(0,p_k,'*');
%         IG = IG + ws(j)* ws(i)*(hs-h);
% %         ylim([0,1]);
% %         ylabel("IG " + round(ws(j)* ws(i)*(hs-h),4), 'FontSize',FONTSIZE);
% %         title("g = "+round(g_bar,3) + " with " +  round(ws(j)* ws(i),4));
%    end 
% end
% disp(IG);
