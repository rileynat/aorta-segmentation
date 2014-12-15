function [ accuracy ] = validate_cnn( val_data, val_labels, weights, filterInfo )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    yhat = forwardFeed_cnn(val_data, val_labels, weights, filterInfo);
    pred = yhat <= 0.5;
    
    for i = 1:size(val_data,3)
        

end

