function [ accuracy ] = validate_cnn( val_data, val_labels, weights, filterInfo )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    [~, theta] = roll_params(0, weights);
    [accuracy, ~] = gradient_cnn(theta, val_data, val_labels, filterInfo, 0);
    

end

