m = 4;
l = 3;
param = em(x, m, l);
plotdensity(x, param, 'b'); hold on;
plotdensity(x, trueparam, 'r'); hold on;