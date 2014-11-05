function [ x ] = fit_HUscale( x )
%FIT_HUSCALE Summary of this function goes here
%   Detailed explanation goes here
x = double(x);
x = x - 1024;
x(find(x<0)) = 0.0;
x(find(x>500)) = 500;
x = x - 250;
%x = bsxfun(@rdivide, bsxfun(@minus, x, normalparams.m), normalparams.stds);
x = x ./ 250;

end

