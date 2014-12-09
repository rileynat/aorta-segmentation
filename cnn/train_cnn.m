function [ weights ] = train_cnn(train_data, train_labels, filterInfo, batchsize )
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
    options.maxiter = numIterations;
    
    num_batches = ceil(size(train_data(3))/batchsize);
    
    start = 0;
    end_ind = 0;
    for batch = 1:num_batches
        start = end_ind + 1;
        end_ind = start + batchsize - 1;
        if end_ind > size(train_data, 3)
            end_ind = size(train_data, 3);
        end

        train_data_batch = train_data(:,:,start:end_ind);
        train_labels_batch = train_labels(:,:,start:end_ind);
        [optTheta, ~] = minFunc(@(p) run_gradient_then_roll(p, train_data_batch, train_labels_batch, filterInfo), theta, options);
        theta = optTheta;
    end

    weights = unroll_params(optTheta, filterInfo, size(train_data,1), size(train_data,2));

    
end

