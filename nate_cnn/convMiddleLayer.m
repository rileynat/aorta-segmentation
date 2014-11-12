function [ conv_mat ] = convMiddleLayer( input, filters, bias, numOutUnits, filterSize, numFilters )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

    batchSize = size(input, 4);
    
    filters_mat = zeros(size(filters));
    for i = 1:numFilters
        for j = 1:numOutUnits
            filters_mat(:,:,i,j) = filters(end:-1:1,end:-1:1,i,j);
        end
    end
    
    convSizeX = size(input, 1) - filterSize + 1;
    convSizeY = size(input, 2) - filterSize + 1;
    
    bias_reshape = reshape(bias, [1 1 numFilters]);
    bias_mat = repmat(bias_reshape, [convSizeX convSizeY 1 batchSize]);
    
    conv_mat = bias_mat;
    
    for i = 1:numFilters
        for j = 1:numOutUnits
            conv_mat(:,:,j,:) = conv_mat(:,:,j,:) + convn(input(:,:,i,:), filters_mat(:,:,i,j), 'valid');
        end
    end
end

