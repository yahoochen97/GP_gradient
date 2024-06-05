function ll = log_mvnpdf(x,mu,K)
    n = numel(mu);
    ll = -n*log(2*pi)/2;
    if det(K)>0
        ll = ll-log(det(K))/2-(x-mu)'*(K\(x-mu))/2;
    else
        jitter = 1e-12*eye(n);
        ll = ll-log(det(K)+jitter)/2-(x-mu)'*((K+jitter)\(x-mu))/2;
    end
end