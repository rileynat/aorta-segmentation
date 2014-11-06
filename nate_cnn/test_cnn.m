function [ accuracy ] = test_cnn( test_data, test_labels, inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias, filterSize, numFilters )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    [cost theta] = roll_params(0, inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias);
    [accuracy, ~, ~, ~, ~] = gradient_cnn(theta, test_data, test_labels, filterSize, numFilters, 0);
    
    

end

