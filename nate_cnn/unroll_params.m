function [ weights ] = unroll_params( theta, filterSize, numFilters, inputXSize, inputYSize )
%unroll_params takes the theta aggregation of all of the learning
%parameters and separates them out
%   Detailed explanation goes here
    weights = struct;
    start = 1;
    size = filterSize * filterSize * numFilters;
    weights.inToHidFilters = reshape(theta(start:start+size-1), [filterSize, filterSize, numFilters]);
    start = start + size;
    size = numFilters;
    weights.inToHidBias = theta(start:start+size-1);
    start = start + size;
    size = filterSize * filterSize * numFilters;
    weights.hidToOutFilters = reshape(theta(start:start+size-1), [filterSize, filterSize, numFilters]);
    start = start + size;
    size = inputXSize * inputYSize;
    weights.hidToOutBias = reshape(theta(start:start+size-1), [inputXSize, inputYSize]);
    
    assert(start + size - 1 == length(theta));

end

