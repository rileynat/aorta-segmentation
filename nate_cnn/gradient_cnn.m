function [ cost, inToHidFilterGrad, inToHidBiasGrad, hidToOutFilterGrad, hidToOutBiasGrad ] = gradient_cnn ( theta, train_data, train_labels, filterSize, numFilters)
%UNTITLED3 Computes the gradient for a single training example
%   Detailed explanation goes here
    
    [ inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias ] = unroll_params(theta, filterSize, numFilters, size(train_data,1), size(train_data,2));

    m = size(train_data, 3);
    
    %forward prop
    hiddenLayerRaw = convolveValid(train_data, inToHidFilters, inToHidBias, filterSize, numFilters);
    hiddenLayer = sigmoid(hiddenLayerRaw);
    outputLayerRaw = convolveFull(hiddenLayer, hidToOutFilters, hidToOutBias, filterSize, numFilters);
    outputLayer = sigmoid(outputLayerRaw);
    
    cost = cost_cnn(outputLayer, train_labels) ./ m;
    
    deltaObj = (outputLayer - train_labels) ./ m;
    
    inToHidFilterGrad = zeros(size(inToHidFilters));
    hidToOutFilterGrad = zeros(size(hidToOutFilters));
    
    hidToOutBiasGrad = sum(deltaObj,3);
    
    hiddenSizeX = size(hiddenLayer,1);
    hiddenSizeY = size(hiddenLayer,2);
    
    for i = 1:numFilters
        conv = convn(deltaObj, reshape(hiddenLayer(end:-1:1, end:-1:1, i, :), [hiddenSizeX hiddenSizeY m]), 'valid'); 
        % Xinchen, why reverse batch?
        hidToOutFilterGrad(:,:,i) = sum(conv,3);
    end
    
    deltaHid = zeros(size(hiddenLayer));
    for i = 1:numFilters
        deltaHid(:,:,i,:) = convn(deltaObj, hidToOutFilters(end:-1:1,end:-1:1,i), 'valid');
    end
    deltaHid = deltaHid .* hiddenLayer .* (1 - hiddenLayer);
    
    deltaSum = sum(sum(sum(deltaHid,1),2),4);
    inToHidBiasGrad = deltaSum(:);
    
    for i = 1:numFilters
        conv = convn(train_data, reshape(deltaHid(:,:,i,:), [hiddenSizeX hiddenSizeY m]), 'valid');
        inToHidFilterGrad(:,:,i) = sum(conv, 3);
    end
    
    
    
    
end
