function normalparams = fit_parameters_HUscale( x )
%FIT_PARAMETERS_HUSCALE Summary of this function goes here
%   Detailed explanation goes here

normalparams = struct('m',[],'stds',[]);

x = double(x);
x = x - 1024;
x(find(x<0)) = 0.0;
x(find(x>500)) = 500;
x = x - 250;
x = x ./ 250;
%normalparams.m = 0.0;
%normalparams.stds = std(x(:));

end

