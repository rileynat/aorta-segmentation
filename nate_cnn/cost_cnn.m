function [ cost ] = cost_cnn( output, labels )
%Computes the error cost of the prediction of the cnn output against the
%training set labels
%   Detailed explanation goes here
    
    output = output(:);
    labels = labels(:);

    % for numerical stability
    output = min(output, 1-1e-8);
    output = max(output, 1e-8);

    cost = -sum(labels.*log(output)+(1-labels).*log(1-output));


end

