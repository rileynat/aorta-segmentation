function [ accuracy ] = validate_cnn( val_data, val_labels, weights, filterSize, numFilters )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [~, theta] = roll_params(0, weights);
    [accuracy, ~] = gradient_cnn(theta, val_data, val_labels, filterSize, numFilters, 0);
    

end

