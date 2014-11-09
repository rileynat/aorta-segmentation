function [ weights ] = train_cnn(train_data, train_labels, numFilters, filterSize )
%train_cnn Takes the normalized input images and outputs the correct
%parameters for training
%   TODO when finished
    rng(1);
    weights = struct;
    weights.inToHidFilters = 0.01 .* randn(filterSize, filterSize, numFilters);
    weights.inToHidBias = zeros(numFilters, 1);
    weights.hidToOutFilters = 0.01 .* randn(filterSize, filterSize, numFilters);
    weights.hidToOutBias = zeros(size(train_data,1), size(train_data,2));
    
    numIterations = 5000;
    
    addpath(genpath('utils/minFunc_2012/'));
    
    [~, theta] = roll_params(0, weights);
    
    % lbfgs
    options.method = 'lbfgs';
    options.maxiter = numIterations;
    
    [optTheta, ~] = minFunc(@(p) run_gradient_then_roll(p, train_data, train_labels, filterSize, numFilters), theta, options);
    weights = unroll_params(optTheta, filterSize, numFilters, size(train_data,1), size(train_data,2));

    
end

