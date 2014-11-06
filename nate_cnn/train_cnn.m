function [ inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias ] = train_cnn(train_data, train_labels, numFilters, filterSize )
%train_cnn Takes the normalized input images and outputs the correct
%parameters for training
%   TODO when finished
    inToHidFilters = 0.01 .* randn(filterSize, filterSize, numFilters);
    inToHidBias = zeros(numFilters, 1);
    hidToOutFilters = 0.01 .* randn(filterSize, filterSize, numFilters);
    hidToOutBias = zeros(size(train_data,1), size(train_data,2));
    numIterations = 500;
    
    addpath(genpath('utils/minFunc_2012/'));
    
    [cost, theta] = roll_params(0, inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias);
    
    % lbfgs
    options.method = 'lbfgs';
    options.maxiter = numIterations;
    
    [optTheta ~] = minFunc(@(p) run_gradient_then_roll(p, train_data, train_labels, filterSize, numFilters), theta, options);
    [inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias] = unroll_params(optTheta, filterSize, numFilters, size(train_data,1), size(train_data,2));

    
end

