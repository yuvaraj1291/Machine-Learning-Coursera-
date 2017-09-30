function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h=sigmoid(X*theta);
logh=log(h);
minus1logh=log(1.-h);
J=sum((-y.*logh) - ((1.-y).*minus1logh));
J= J./m ;

regtheta=theta(2:end);
regtheta=regtheta.^2;
reg=sum(regtheta);
reg=(reg*lambda)/(2*m);

J=J+reg;

diff=h-y;

grad=diff.*X;
grad=sum(grad);
grad=grad./m;

reggrad= (lambda.*theta(2:end))./m;
reggrad=[zeros(1);reggrad];
grad=grad'+reggrad;

% =============================================================

end
