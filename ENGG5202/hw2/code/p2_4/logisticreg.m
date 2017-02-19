% ----------------------------------------
% function [w] = logisticreg(X,y,c)
%
% Estimates logistic regression coefficients
% with Newton-Raphson 
%
% each row of X is a training example 
% y is a column vector of (0/1) labels
% c is the regularization parameter

function [w] = logisticreg(X,y,c) 

if (nargin < 3)
  % small regularization
  c = 1e-4;
end

X = [ones(size(X,1),1),X];
[n,d] = size(X);

w = zeros(d,1);

cont = 1;
while (cont),
    p = g(X*w); % P(y=1|x,w)

    Dw  = ((y-p)'*X)'-c*w;  % gradient 

    Z = repmat(p.*(1-p),1,d).*X;
    DDw = -(c*eye(d) + Z'*X); % Hessian (second derivatives)

    wo = w;
    w = wo - inv(DDw)*Dw; % Newton-Raphson step

    cont = norm(w-wo)>1e-6; % stopping criterion
end;
    
% ----------------------------------------
% logistic function

function [p] = g(z)

p = 1 ./(1+exp(-z));

