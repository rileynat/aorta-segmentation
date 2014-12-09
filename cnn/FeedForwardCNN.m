function [ outputLayer ] = FeedForwardCNN( data, weights, filterInfo )
% Performs a feed forward pass of |data| in the CNN with structure
% specified by filterInfo using the weights specified by |weights|.

    hiddenLayer1Raw = convFirstLayer(data, weights.inToHidFilters, weights.inToHidBias, filterInfo.filterSize1, filterInfo.numFilters1);
    hiddenLayer1 = sigmoid(hiddenLayer1Raw);
    hiddenLayer2Raw = convMiddleLayer(hiddenLayer1, weights.hidToHidFilters, weights.hidToHidBias, filterInfo.numFilters2, filterInfo.filterSize2, filterInfo.numFilters1);
    hiddenLayer2 = sigmoid(hiddenLayer2Raw);
    outputLayerRaw = convFinalLayer(hiddenLayer2, weights.hidToOutFilters, weights.hidToOutBias, filterInfo.filterSize3, filterInfo.numFilters3);
    outputLayer = sigmoid(outputLayerRaw);

end

