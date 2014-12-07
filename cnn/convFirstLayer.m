function [ conv_mat ] = convFirstLayer( input_data, filters, bias, filterSize, numFilters )
%convFirstLayer Computest the convolution of the two matrices with type
%valid for the first layer
%   Detailed explanation goes here

    batchSize = size(input_data, 3);

    filters_mat = zeros(size(filters));
    for i = 1:numFilters
        filters_mat(:,:,i) = filters(end:-1:1,end:-1:1,i);
    end
    
    convSizeX = size(input_data, 1) - filterSize + 1;
    convSizeY = size(input_data, 2) - filterSize + 1;
    
    bias_reshape = reshape(bias, [1 1 numFilters]);
    bias_mat = repmat(bias_reshape, [convSizeX convSizeY 1 batchSize]);
    
    conv_mat = bias_mat;

    
    for i = 1:numFilters
       
       conv = convn(input_data(:,:,:), filters_mat(:,:,i), 'valid');
       conv_mat(:,:,i,:) = conv_mat(:,:,i,:) + permute(conv, [1 2 4 3]);
    end
    

end

