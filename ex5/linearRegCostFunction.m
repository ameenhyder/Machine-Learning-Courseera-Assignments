function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0; %#ok<*NASGU> 
grad = zeros(size(theta)); %#ok<*PREALL> 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

H = X * (theta);
J =  (sum((H - y) .^ 2))/(2*m);
J_reg = ((sum((theta(2:end,:) .^ 2)))*lambda)/(2*m);
J = J + J_reg;

grad = (((H - y)' * X)/m)';
grad_reg = zeros(size(grad));
grad_reg(2:end,:) = (theta(2:end,:) .* lambda)/m;
grad = grad + grad_reg;











% =========================================================================

grad = grad(:);

end
