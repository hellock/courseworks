
function [w] = fisherdiscriminant(X,y) 

class0 = find(y==0);
class1 = find(y==1);

mu0 = mean(X(class0,:))';
mu1 = mean(X(class1,:))';

cov0 = cov(X(class0,:));
cov1 = cov(X(class1,:));

n0 = length(class0);
n1 = length(class1);

w = inv(n0*cov0 + n1*cov1)*(mu1-mu0);
w = [-w'*(mu0 + mu1)/2 + 0.5 * log(n1/n0); w];