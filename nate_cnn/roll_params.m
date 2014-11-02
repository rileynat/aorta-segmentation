function [ cost, theta ] = roll_params( cost, inToHidFilters, inToHidBias, hidToOutFilters, hidToOutBias )
%roll_params transforms the 4 different weight parameters into a single
%long vector for minFunc
%   Detailed explanation goes here

    cost = cost;
    theta = [];
    theta = [theta ; inToHidFilters(:)];
    theta = [theta ; inToHidBias(:)];
    theta = [theta ; hidToOutFilters(:)];
    theta = [theta ; hidToOutBias(:)];

end

