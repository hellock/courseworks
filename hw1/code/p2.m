%%
x = -2:0.01:3;
y = exp(-x);
y(x<0) = 0;
plot(x, y);
xlabel('x');
ylabel('p(x|\theta)');
%%
theta = 0:0.01:5;
y = theta .* exp(-2 * theta);
plot(theta, y);
xlabel('\theta');
ylabel('p(x|\theta)');
