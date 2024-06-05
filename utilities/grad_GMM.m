function [ws,mu_GMM,sigma_GMM] = grad_GMM(n_gauss_hermite,hyp,inffunc,meanfunc, likfunc, x, y, xs) 
% Approximate dp(y=1)/dx with GMM with SE kernel
% 
% Copyright (c) by Yehu Chen, 2023-02-22.
%
    N = size(xs,1);
    covfunc = {@covSEiso};
    [~,~,fmu,fs2, ~, post] = gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs);

    % squared exponential kernel first gradient
    dK = feval(covfunc{:},hyp.cov,x,xs)/exp(2*hyp.cov(1));
    dK = dK.*repmat(x,1,numel(xs)) - dK.*repmat(xs',numel(x),1);

    % squared exponential kernel second gradient
    d2K = feval(covfunc{:},hyp.cov,xs,xs)/exp(2*hyp.cov(1)).*(ones(numel(xs),numel(xs))...
        -sq_dist(xs,xs)/exp(2*hyp.cov(1)));

    % gradient gp posterior mean
    dm = dK'*post.alpha; % + hyp.mean(2) + 2*hyp.mean(3)*xs;

    % gradient gp posterior cov
    tmp = post.L'\(repmat(post.sW,1,numel(xs)).*dK);
    ds2 = d2K - tmp'*tmp;

    % get Gauss-Hermite weights / points
    [ks,ws] = root_GH(n_gauss_hermite);

    mu_GMM = {};
    sigma_GMM = {};
    for i=1:n_gauss_hermite
        f_bar = sqrt(2)*ks(i)*sqrt(fs2) + fmu;
        mu_bar = normpdf(f_bar).*dm; % mu_bar = dm;
        K_bar = (normpdf(f_bar)*normpdf(f_bar)').*ds2; % K_bar = ds2;
        mu_GMM{i} = mu_bar;
        sigma_GMM{i} = sqrt(diag(K_bar));
    end
end