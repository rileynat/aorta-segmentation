function [ weights ] = unroll_params( theta, filterInfo, inputXSize, inputYSize )
%unroll_params takes the theta aggregation of all of the learning
%parameters and separates them out
%   Detailed explanation goes here
    weights = struct;
    start = 1;
    size = filterInfo.filterSize1 * filterInfo.filterSize1 * filterInfo.numFilters1;
    weights.inToHidFilters = reshape(theta(start:start+size-1), [filterInfo.filterSize1, filterInfo.filterSize1, filterInfo.numFilters1]);
    start = start + size;
    size = filterInfo.numFilters1;
    weights.inToHidBias = theta(start:start+size-1);
    start = start + size;
    size = filterInfo.filterSize2 * filterInfo.filterSize2 * filterInfo.numFilters1 * filterInfo.numFilters2;
    weights.hidToHidFilters = reshape(theta(start:start+size-1), [filterInfo.filterSize2, filterInfo.filterSize2, filterInfo.numFilters1, filterInfo.numFilters2]);
    start = start + size;
    size = filterInfo.numFilters2;
    weights.hidToHidBias = theta(start:start+size-1);
    start = start + size;
    size = filterInfo.filterSize3 * filterInfo.filterSize3 * filterInfo.numFilters3;
    weights.hidToOutFilters = reshape(theta(start:start+size-1), [filterInfo.filterSize3, filterInfo.filterSize3, filterInfo.numFilters3]);
    start = start + size;
    size = inputXSize * inputYSize;
    weights.hidToOutBias = reshape(theta(start:start+size-1), [inputXSize, inputYSize]);
    
    assert(start + size - 1 == length(theta));

end

