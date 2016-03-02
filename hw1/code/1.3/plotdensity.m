
function [] = plotdensity(x,param,st)

if (nargin<3), st = 'r'; end;

n = 500;

xmin = min(param.mu)-3*max(param.v);
xmax = max(param.mu)+3*max(param.v);
xp = xmin + (0:n-1)'/(n-1)*(xmax-xmin);

m = length(param.mu);
l = length(param.v);

p = zeros(size(xp));
for j=1:m,
  for k=1:l,
    p=p+param.p(j)*param.q(k)*normpdf(xp,param.mu(j),sqrt(param.v(k)));
  end;
end;

plot(xp,p,st); hold on;

if (nargin>1), 
  plot(x,zeros(size(x)),'k*');
end;

hold off;

