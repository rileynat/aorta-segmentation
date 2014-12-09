function [features] = FeedForwardAutoencoder(theta, saeInfo, data)

W1 = reshape(theta(1:saeInfo.hiddenSize*saeInfo.inputSize), ...
    saeInfo.hiddenSize, saeInfo.inputSize);
b1 = theta(2*saeInfo.hiddenSize*saeInfo.inputSize+ ... 
    1:2*saeInfo.hiddenSize*saeInfo.inputSize+saeInfo.hiddenSize);

if size(W1, 1) ~= size(b1, 1)
    b1 = b1(2:end);
end

z2 = bsxfun(@plus, b1, W1*data);
a2 = sigmoid(z2);
features = a2;

end
