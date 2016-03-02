function [p] = softmax(logp);

[n,m] = size(logp);
logmax = max(logp,[],2);
logp = logp-logmax*ones(1,m);
p = exp(logp);
pnorm = sum(p,2);
p = p ./(pnorm*ones(1,m));

