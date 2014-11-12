function [ cost, grad_vec ] = run_gradient_then_roll( theta, train_data, train_labels, filterInfo )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [cost, grad] = gradient_cnn(theta, train_data, train_labels, filterInfo, 1);
    [cost, grad_vec] = roll_params(cost, grad);
    %numGrad = gradientChecking( @(x) gradient_cnn(x, train_data, train_labels, filterInfo, 0), theta, grad_vec);
    
end

