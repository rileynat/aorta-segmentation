function [ cost, theta ] = roll_params( cost, weights )
%roll_params transforms the 4 different weight parameters into a single
%long vector for minFunc
%   Detailed explanation goes here

    cost = cost;
    theta = [];
    theta = [theta ; weights.inToHidFilters(:)];
    theta = [theta ; weights.inToHidBias(:)];
    theta = [theta ; weights.hidToHidFilters(:)];
    theta = [theta ; weights.hidToHidBias(:)];
    theta = [theta ; weights.hidToOutFilters(:)];
    theta = [theta ; weights.hidToOutBias(:)];

end

