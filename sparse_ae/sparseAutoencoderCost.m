function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

if size(W1, 1) ~= size(b1, 1)
    b1 = b1(2:end);
end

if size(W2, 1) ~= size(b2, 1)
    b2 = b2(2:end);
end

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 

% Get num training examples
m = size(data, 2);

%% Compute activations for each layer
z2 = bsxfun(@plus, b1, W1*data);
a2 = sigmoid(z2);
z3 = bsxfun(@plus, b2, W2*a2);
h = sigmoid(z3); % a3

c_init = sum(sum((h - data).^2)) / (2 * m);
reg = (sum(sum(W1.^2)) + sum(sum(W2.^2)));
rho_hat = sum(a2, 2) / m;
rho = sparsityParam;
sparse_left = rho .* log(rho ./ rho_hat);
sparse_right = (1 - rho) .* log((1 - rho) ./ (1 - rho_hat));
sparse_pen = sum(sparse_left + sparse_right);

cost = c_init + reg * .5 * lambda + beta * sparse_pen; %%switch with cost once it works
%cost = c_init + reg * .5 * lambda;

d3 = -(data - h) .* sigmoid_deriv(z3);
%d2_sparse = beta .* ((-sparsityParam ./ rho) + (1 - sparsityParam) ./ (1 - rho));

d2_penalty = -(rho./rho_hat) + (1-rho) ./ (1-rho_hat);

d2 = bsxfun(@plus, W2' * d3,  beta * d2_penalty) .* sigmoid_deriv(z2);  %%switch this with d2 once it works
%d2 = (W2' * d3) .* sigmoid_deriv(z2);

W2grad = (gpuArray(d3) * gpuArray(a2')) / gpuArray(m) + gpuArray(lambda) * gpuArray(W2);

W1grad = (gpuArray(d2) * gpuArray(data')) /gpuArray(m) + gpuArray(lambda) * gpuArray(W1);

b1grad = (1 / m) .* sum(d2, 2);

b2grad = (1 / m) .* sum(d3, 2);



%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [gather(W1grad(:)) ; gather(W2grad(:)) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

% f'
function sigm_d = sigmoid_deriv(x)

	sigm_d = sigmoid(x) .* (1 - sigmoid(x));
end

function c = compute_cost(hyp, y)

	

end

