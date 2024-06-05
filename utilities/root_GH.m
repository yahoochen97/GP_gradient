function [ks, ws]=root_GH(N)
% compute roots of the N-th physicist's hermite polys
% and corresponding weights for Gauss-Hermite quadrature
    p = HermitePoly(N);
    ks = roots(p);
    ks = sort(ks);
    ws = 2^(N-1)*factorial(N)./(N*polyval(HermitePoly(N-1),ks)).^2;
end

% HermitePoly.m by David Terr, Raytheon, 5-10-04
% Given nonnegative integer n, compute the 
% Hermite polynomial H_n. Return the result as a vector whose mth
% element is the coefficient of x^(n+1-m).
% polyval(HermitePoly(n),x) evaluates H_n(x).
function hk = HermitePoly(n)
    if n==0 
        hk = 1;
    elseif n==1
        hk = [2 0];
    else
    end

    hkm2 = zeros(1,n+1);
    hkm2(n+1) = 1;
    hkm1 = zeros(1,n+1);
    hkm1(n) = 2;
    for k=2:n

        hk = zeros(1,n+1);
        for e=n-k+1:2:n
            hk(e) = 2*(hkm1(e+1) - (k-1)*hkm2(e));
        end

        hk(n+1) = -2*(k-1)*hkm2(n+1);

        if k<n
            hkm2 = hkm1;
            hkm1 = hk;
        end

    end
end