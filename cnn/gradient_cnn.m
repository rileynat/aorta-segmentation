function [ cost, grad ] = gradient_cnn ( theta, train_data, train_labels, filterInfo, gradFlag)
%UNTITLED3 Computes the gradient for a single training example
%   Detailed explanation goes here
    
    weights = unroll_params(theta, filterInfo, size(train_data,1), size(train_data,2));

    m = size(train_data, 3);
    
    %forward prop
    hiddenLayer1Raw = convFirstLayer(train_data, weights.inToHidFilters, weights.inToHidBias, filterInfo.filterSize1, filterInfo.numFilters1);
    hiddenLayer1 = sigmoid(hiddenLayer1Raw);
    hiddenLayer2Raw = convMiddleLayer(hiddenLayer1, weights.hidToHidFilters, weights.hidToHidBias, filterInfo.numFilters2, filterInfo.filterSize2, filterInfo.numFilters1);
    hiddenLayer2 = sigmoid(hiddenLayer2Raw);
    outputLayerRaw = convFinalLayer(hiddenLayer2, weights.hidToOutFilters, weights.hidToOutBias, filterInfo.filterSize3, filterInfo.numFilters3);
    outputLayer = sigmoid(outputLayerRaw);
    
    cost = cost_cnn(outputLayer, train_labels) ./ m;
    
    %backprop
    deltaObj = (outputLayer - train_labels) ./ m;
    
    grad = struct;
    grad.inToHidFilters = zeros(size(weights.inToHidFilters));
    grad.hidToHidFilters = zeros(size(weights.hidToHidFilters));
    grad.hidToOutFilters = zeros(size(weights.hidToOutFilters));
    grad.inToHidBias = zeros(size(weights.inToHidBias));
    grad.hidToHidBias = zeros(size(weights.hidToHidBias));
    
    grad.hidToOutBias = sum(deltaObj,3);
    
    if gradFlag == 1

        for i = 1:filterInfo.numFilters1
            %why reverse the filters
            grad.hidToOutFilters(:,:,i) = convn(deltaObj, permute(hiddenLayer2(end:-1:1, end:-1:1, i, end:-1:1), [1 2 4 3]), 'valid'); 
        end
        
        deltaHid2 = zeros(size(hiddenLayer2));
        for i = 1:filterInfo.numFilters2
            deltaHid2(:,:,i,:) = convn(deltaObj, weights.hidToOutFilters(end:-1:1, end:-1:1, i), 'valid');
        end
        deltaHid2 = deltaHid2 .* hiddenLayer2 .* (1 - hiddenLayer2);

        deltaSum = sum(sum(sum(deltaHid2,1), 2), 4);
        grad.hidToHidBias = deltaSum(:);
        
        for i = 1:filterInfo.numFilters1
            for j = 1:filterInfo.numFilters2
                grad.hidToHidFilters(:,:,i,j) = convn(hiddenLayer1(:,:,i,:), deltaHid2(end:-1:1, end:-1:1, j, end:-1:1), 'valid');
            end
        end
        
        deltaHid1 = zeros(size(hiddenLayer1));
        for i = 1:filterInfo.numFilters1
            for j = 1:filterInfo.numFilters2
                %deltaHid1(:,:,i,:) = deltaHid1(:,:,i,:) + convn(weights.hidToHidFilters(:,:,i,j), deltaHid2(:,:,j,:), 'full');
                deltaHid1(:,:,i,:) = deltaHid1(:,:,i,:) + convn(deltaHid2(:,:,j,:), weights.hidToHidFilters(:,:,i,j), 'full');
            end
        end
        deltaHid1 = deltaHid1 .* hiddenLayer1 .* (1 - hiddenLayer1);
        
        deltaSum = sum(sum(sum(deltaHid1,1), 2), 4);
        grad.inToHidBias = deltaSum(:);
        for i = 1:filterInfo.numFilters1
            grad.inToHidFilters(:,:,i) = convn(train_data, permute(deltaHid1(end:-1:1, end:-1:1, i, end:-1:1), [1 2 4 3]), 'valid');
        end
    end
    
    
    
end

