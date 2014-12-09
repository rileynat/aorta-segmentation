function [ conv_mat ] = convFinalLayer( input, filters, bias, filterSize, numFilters )
%convFinalLayer returns the full convolution of the input with the filters
%and bias
%   Detailed explanation goes here

    batchSize = size(input, 4);
    
    bias_mat = repmat(bias, [1 1 1 batchSize]);
    
    conv_mat = bias_mat;

    for i = 1:numFilters
        conv_mat = conv_mat + convn(input(:,:,i,:), filters(:,:,i), 'full');
    end
    
    conv_mat = permute(conv_mat, [1 2 4 3]);
    
end

