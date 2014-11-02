function [ cost, theta ] = run_gradient_then_roll( theta, train_data, train_labels, filterSize, numFilters )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [cost, inToHidFilterGrad, inToHidBiasGrad, hidToOutFilterGrad, hidToOutBiasGrad] = gradient_cnn(theta, train_data, train_labels, filterSize, numFilters);
    [cost, theta] = roll_params(cost, inToHidFilterGrad, inToHidBiasGrad, hidToOutFilterGrad, hidToOutBiasGrad);

end

