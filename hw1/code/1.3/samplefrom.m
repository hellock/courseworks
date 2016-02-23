
function [x] = samplefrom(param,n)

m = length(param.mu);
l = length(param.v);

pcum = cumsum(param.p);
qcum = cumsum(param.q);

x = zeros(n,1);
for i=1:n,
  r = rand; j = min(find(r<=pcum));
  r = rand; k = min(find(r<=qcum));
  x(i) = randn*sqrt(param.v(k))+param.mu(j);
end;



