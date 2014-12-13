function [ weights ] = train_cnn(train_data, train_labels, filterInfo)
%train_cnn Takes the normalized input images and outputs the correct
%parameters for training
%   TODO when finished

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
    options.maxiter = 500;
    
    batchsize = floor(size(train_data, 3)/4);
%    train_x_1 = train_data(:, :, 1:batchsize);
%    train_y_1 = train_labels(:, :, 1:batchsize);
%    train_x_2 = train_data(:, :, batchsize+1:2*batchsize);
%    train_y_2 = train_labels(:, :, batchsize+1:2*batchsize);
%    train_x_3 = train_data(:, :, 2*batchsize+1:3*batchsize);
%    train_y_3 = train_labels(:, :, 2*batchsize+1:3*batchsize);
%    train_x_4 = train_data(:, :, 3*batchsize+1:end); 
%    train_y_4 = train_labels(:, :, 3*batchsize+1:end); 
    
    
%    for iter= 1:1000
        [optTheta, ~] = minFunc(@(p) run_gradient_then_roll(p, train_data, train_labels, filterInfo), theta, options);
        theta = optTheta;


%    end

    weights = unroll_params(optTheta, filterInfo, size(train_data,1), size(train_data,2));

    
end

