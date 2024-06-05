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
fig=figure(1); tiledlayout(1,2);

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
title("GMM approximation of dp(y=1)/dx", 'FontSize', FONTSIZE);

% empirical sample f and df/dx
nexttile;

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

title("numerical samples from dp(y=1)/dx", 'FontSize', FONTSIZE);
filename = "./results/GMM_example.pdf";
set(fig, 'PaperPosition', [-1 -0.05 19 3]); 
set(fig, 'PaperSize', [17 2.8]);
print(fig, filename, '-dpdf','-r300');
close;

% acquisition UCB, DE-U, BALD
hyp.mean = [-1;0;1];
hyp.cov = [log(1);log(2)];
n = 3;
x = [-1.8;min(max(normrnd(-1.0,0.25,n,1),-2),2);...
    min(max(normrnd(0,0.25,n,1),-2),2);...
    min(max(normrnd(1.5,0.1,n,1),-2),2);1.8];
f = 4*x.^2-2;
p = normcdf(f);
y = 2*binornd(1,p)-1;

N = 101;
xs = linspace(-2,2,N)';
n_gauss_hermite = 5;

% visualize data
fig=figure(2); tiledlayout(2,2);

[~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);
nexttile;
plot(xs, normcdf(fmu./sqrt(1+fs2)), 'LineWidth',2, 'color', [40, 150, 215] / 255); hold on;
f = [normcdf(fmu+2*sqrt(fs2)); flipdim(normcdf(fmu-2*sqrt(fs2)),1)];
% f = [(ymu+1)/2+2*sqrt(ys2); (ymu+1)/2-2*sqrt(ys2)];
fill([xs; flipdim(xs,1)], f, [80, 200, 245] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none');
scatter(x(y==1),y(y==1), 60, 'r*');
scatter(x(y==-1),y(y==-1)+1, 60, 'k*');
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


% DE-ME
nexttile;
IG_p = abs(fmu+1.96*sqrt(fs2));
IG_p = IG_p / max(IG_p) / 1.2;
f = [IG_p; xs*0];

plot(xs, IG_p, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx] = max(IG_p);
scatter(xs(idx),IG_p(idx)+0.04,40,'v','MarkerFaceColor',[0 0 0]);
ylim([0,1]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["{\alpha}(x;D)", "new x = " + xs(idx)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel('UCB', 'FontSize',FONTSIZE);

% mutual information I(y,p;x,D): BALD
nexttile;
ps = normcdf(fmu./sqrt(1+fs2));
C = sqrt(pi*log(2)/2);
IG_p = -ps.*log2(ps) - (1-ps).*log2(1-ps) - ...
    C./sqrt(C^2+fs2).*exp(-fmu.^2./(C^2+fs2)/2);
f = [IG_p; xs*0];

plot(xs, IG_p, 'color', [250,127,32]/255, 'LineWidth', 1.5); hold on;
[~, idx] = max(IG_p);
scatter(xs(idx),IG_p(idx)+0.02,40,'v','MarkerFaceColor',[0 0 0]);
ylim([0,0.45]);
fill([xs; flipdim(xs,1)], f, [253,183,80] / 255, ...
     'facealpha', 0.5, ...
     'edgecolor', 'none'); 

legend(["{\alpha}(x;D)", "new x = " + xs(idx)],... 
    'Location', 'northwest','NumColumns',2, 'FontSize',FONTSIZE);
legend('boxoff');
ylabel('BALD', 'FontSize',FONTSIZE);


filename = "./results/acquisition_example.pdf";
% set(fig, 'PaperPosition', [-0.7 -0.1 25 3]); 
% set(fig, 'PaperSize', [23 2.8]);
set(fig, 'PaperPosition', [-0.7 -0.3 25 6]); 
set(fig, 'PaperSize', [23 5.4]);
print(fig, filename, '-dpdf','-r300');
close;
