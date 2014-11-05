function [ yrefined ] = gaussian_filter( yraw, filterparams )
%GAUSSIAN_FILTER Summary of this function goes here
%   Detailed explanation goes here

winsize = filterparams.winsize;
radius = (winsize - 1)/2;

gkernel = zeros(filterparams.winsize, 1);
for k = -radius:radius
    gkernel(k+radius+1, 1) = exp(-abs(k).^2*(filterparams.lambda.^2));
end
gkernel = gkernel ./ sum(gkernel);
y = reshape(yraw, [size(yraw,1)*size(yraw,2), size(yraw, 3)]); %% [y_1;y_2;...;y_T]
yrefined = conv2(y, gkernel', 'same');
for k = 1:radius
    yrefined(:,k) = yrefined(:,k) ./ sum(gkernel(1:k+radius));
    yrefined(:,size(yraw, 3)-k+1) = yrefined(:,size(yraw,3)-k+1) ./ sum(gkernel(1:k+radius));
end

yrefined = reshape(yrefined, [size(yraw, 1) size(yraw, 2) size(yraw, 3)]);

end

