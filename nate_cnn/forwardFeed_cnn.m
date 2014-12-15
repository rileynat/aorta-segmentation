function [ yhat, cost ] = forwardFeed_cnn( input, labels, weights, filterInfo )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    hiddenLayer = sigmoid(convFirstLayer(input, weights.inToHidFilters, weights.inToHidBias, filterInfo.filterSize1, filterInfo.numFilters1));
    hiddenLayer2 = sigmoid(convMiddleLayer(hiddenLayer, weights.hidToHidFilters, weights.hidToHidBias, filterInfo.numFilters2, filterInfo.filterSize2, filterInfo.numFilters1));
    yhat = sigmoid(convFinalLayer(hiddenLayer2, weights.hidToOutFilters, weights.hidToOutBias, filterInfo.filterSize3, filterInfo.numFilters3));
    
    cost = cost_cnn(yhat, labels) ./ size(input,3);
end

