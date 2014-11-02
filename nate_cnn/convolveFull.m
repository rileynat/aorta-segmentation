function [ conv_mat ] = convolveFull( input, filters, bias, filterSize, numFilters )
%ConvolveFull returns the full convolution of the input with the filters
%and bias
%   Detailed explanation goes here

    batchSize = size(input, 4);
    
    filters_mat = zeros(size(filters));
    for i = 1:numFilters
        filters_mat(:,:,i) = filters(end:-1:1,end:-1:1,i);
    end

    convSizeX = size(input, 1) + filterSize - 1;
    convSizeY = size(input, 2) + filterSize -1;
    
    bias_mat = repmat(bias, [1 1 1 batchSize]);
    
    conv_mat = zeros(convSizeX, convSizeY, numFilters, batchSize);
    
    for i = 1:numFilters
        conv_mat(:,:,i,:) = convn(input(:,:,i,:), filters_mat(:,:,i), 'full');
    end
    
    conv_mat = sum(conv_mat, 3);
    conv_mat = conv_mat + bias_mat;
    conv_mat = reshape(conv_mat, [convSizeX convSizeY batchSize]);
    
end
