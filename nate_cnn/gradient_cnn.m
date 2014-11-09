function [ cost, grad ] = gradient_cnn ( theta, train_data, train_labels, filterSize, numFilters, gradFlag)
%UNTITLED3 Computes the gradient for a single training example
%   Detailed explanation goes here
    
    weights = unroll_params(theta, filterSize, numFilters, size(train_data,1), size(train_data,2));

    m = size(train_data, 3);
    
    %forward prop
    hiddenLayerRaw = convFirstLayer(train_data, weights.inToHidFilters, weights.inToHidBias, filterSize, numFilters);
    hiddenLayer = sigmoid(hiddenLayerRaw);
    outputLayerRaw = convFinalLayer(hiddenLayer, weights.hidToOutFilters, weights.hidToOutBias, filterSize, numFilters);
    outputLayer = sigmoid(outputLayerRaw);
    
    cost = cost_cnn(outputLayer, train_labels) ./ m;
    
    %backprop
    deltaObj = (outputLayer - train_labels) ./ m;
    
    grad = struct;
    grad.inToHidFilters = zeros(size(weights.inToHidFilters));
    grad.hidToOutFilters = zeros(size(weights.hidToOutFilters));
    
    grad.hidToOutBias = sum(deltaObj,3);
    grad.inToHidBias = zeros(size(weights.inToHidBias));
    if gradFlag == 1

        for i = 1:numFilters
            %why reverse the filters
            grad.hidToOutFilters(:,:,i) = convn(deltaObj, permute(hiddenLayer(end:-1:1, end:-1:1, i, end:-1:1), [1 2 4 3]), 'valid'); 
        end

        deltaHid = zeros(size(hiddenLayer));
        for i = 1:numFilters
            deltaHid(:,:,i,:) = convn(deltaObj, weights.hidToOutFilters(end:-1:1, end:-1:1, i), 'valid');
        end
        deltaHid = deltaHid .* hiddenLayer .* (1 - hiddenLayer);

        deltaSum = sum(sum(sum(deltaHid,1),2),4);
        grad.inToHidBias = deltaSum(:);

        for i = 1:numFilters 
            %why reverse the filters
            grad.inToHidFilters(:,:,i) = convn(train_data, permute(deltaHid(end:-1:1, end:-1:1, i, end:-1:1), [1 2 4 3]), 'valid');
        end
    end
    
    
    
end

