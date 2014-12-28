function [ weights ] = train_cnn(train_data, train_labels, filterInfo )
%train_cnn Takes the normalized input images and outputs the correct
%parameters for training
%   TODO when finished
    rng(1);

    weights = struct;
    weights.inToHidFilters = 0.01 .* randn(filterInfo.filterSize1, filterInfo.filterSize1, filterInfo.numFilters1);
    weights.inToHidBias = zeros(filterInfo.numFilters1, 1);
    weights.hidToHidFilters = 0.01 .* randn(filterInfo.filterSize2, filterInfo.filterSize2, filterInfo.numFilters1, filterInfo.numFilters2);
    weights.hidToHidBias = zeros(filterInfo.numFilters2, 1);
    weights.hidToOutFilters = 0.01 .* randn(filterInfo.filterSize3, filterInfo.filterSize3, filterInfo.numFilters3);
    weights.hidToOutBias = zeros(size(train_data,1), size(train_data,2));
    
    numIterations = 5000;
    
    addpath(genpath('utils/minFunc_2012/'));
    
    [~, theta] = roll_params(0, weights);
    
    % lbfgs
    options.method = 'lbfgs';
    options.maxiter = numIterations;
    options.useMex = 0;
    
    [optTheta, ~] = minFunc(@(p) run_gradient_then_roll(p, train_data, train_labels, filterInfo), theta, options);
    weights = unroll_params(optTheta, filterInfo, size(train_data,1), size(train_data,2));

    
end

