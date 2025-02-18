function [pdq,dq]=dqtest(y,f,a,lag)

n=length(y);
hits=(y<f).*(1-a);
hits=hits+(y>f).*(-a);

q=2+lag;

if sum(y<f)>0

ns=n-lag;

xmat=[ones(ns,1) f(lag+1:n)];
for k=1:lag
    lk=lag-k+1;
    xmat=[xmat hits(lk:n-k)];
end
hx=hits((lag+1):n)'*xmat;
xtx=(xmat'*xmat)\eye(q);
%dq=hx*xtx*hx'/n;
dq=hx*xtx*hx';
dq=dq/(a*(1-a));

pdq=1-chi2cdf(dq,q);

else
    %pdq=0.05*rand;
    %dq=chi2inv(pdq,q);
    pdq=NaN;
    dq=NaN;
% 0 violations, thus do binomial test
%   n=length(y);x=0;
%   pdq=binocdf(0,n,a);
%   dq=chi2inv(1-pdq,q);
    
end

