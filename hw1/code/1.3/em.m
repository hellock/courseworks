
function [param] = em(x,m,l,pcounts)

if (nargin<4), pcounts = 0.01; end;

param = initialize(x,m,l,pcounts); 
[param,L] = one_em_iter(x,param);

L0 = L-abs(L); 
while (L-L0>abs(L)*1e-6),
  L0 = L;
  [param,L] = one_em_iter(x,param);
end;

param.loglik = L; 

% --------------------------------------------
function [param] = initialize(x,m,l,pcounts)

n = size(x,1);

[xr,I] = sort(rand(n,1));
param.mu = zeros(m,1);
for j=1:m, param.mu(j) = x(I(j)); end;
param.p = ones(m,1)/m;

v = var(x);
param.v = zeros(l,1);
for k=1:l, param.v(k) = v/2^k; end;
param.z  = v; % prior variance
param.nz = pcounts; % equivalent sample size

param.q = ones(l,1)/l;

% --------------------------------------------
function [param,loglik] = one_em_iter(x,param);

m = length(param.p);
l = length(param.q);

x = reshape(x,length(x),1);
logp = zeros(size(x,1),m*l);
ind = 1;
for j=1:m,
  for k=1:l,
    logp(:,ind)=evalgauss(x,param.mu(j),param.v(k))+log(param.p(j)*param.q(k));
    ind = ind+1;
  end;
end;

logpmax = max(logp,[],2);

% we also want to return the log-likelihood of the data
loglik = sum(logpmax + log(sum(exp(logp-logpmax*ones(1,m*l)),2)));
loglik = loglik + sum(evalprior(param.v,param.z,param.nz));

pos = softmax(logp); % posterior assignments 

% solve for p and q

param.p = sum(reshape(sum(pos,1),l,m),1)';
param.p = param.p/sum(param.p);

param.q = sum(reshape(sum(pos,1),l,m),2);
param.q = param.q/sum(param.q);

% solve for the means (fixed variances)

vi = 1./param.v; 
Wx = vi'*reshape(x'*pos,l,m); 
W1 = vi'*reshape(sum(pos,1),l,m); 
param.mu = Wx'./W1';

% solve for the variances (fixed means)

S = zeros(l,1);
ind = 1;
for j=1:m,
  for k=1:l,
    S(k) = S(k) + pos(:,ind)'*(x-param.mu(j)).^2;
    ind = ind + 1;
  end;
end;

ntot = sum(reshape(sum(pos,1),l,m),2);
param.v = (S+param.nz*param.z)./(ntot+param.nz);

% --------------------------------------------
function [loglik] = evalprior(v,z,nz)

loglik = -0.5*nz*( z./v + log(v) );


% --------------------------------------------
function [loglik] = evalgauss(x,mu,v)

loglik = -0.5*(x-mu).^2/v - 0.5*log(2*pi*v);

