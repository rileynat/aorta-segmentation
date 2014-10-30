function [ yrefined ] = laplacian_filter(yraw, filterparams)
%LAPLACIAN_FILTER Summary of this function goes here
%   Detailed explanation goes here

slow = 0;
winsize = filterparams.winsize;
radius = (winsize - 1)/2;

if slow,
    yrefined = zeros(size(yraw));
    seqL = size(yraw, 3);
    for i = 1:seqL,
        ratios = 0.0;
        for k = max(1, i-radius):min(seqL, i+radius),
            cur_r = exp(-abs(k-i)*filterparams.lambda);
            ratios = ratios + cur_r;
            yrefined(:,:,i) = yrefined(:,:,i) + cur_r .* yraw(:,:,k);
        end
        yrefined(:,:,i) = yrefined(:,:,i) ./ ratios;
    end
else
    
    lkernel = zeros(filterparams.winsize, 1);
    for k = -radius:radius
        lkernel(k+radius+1, 1) = exp(-abs(k)*filterparams.lambda);
    end
    lkernel = lkernel ./ sum(lkernel);
    y = reshape(yraw, [size(yraw,1)*size(yraw,2), size(yraw, 3)]); %% [y_1;y_2;...;y_T]
    yrefined = conv2(y, lkernel', 'same');
    for k = 1:radius
        yrefined(:,k) = yrefined(:,k) ./ sum(lkernel(1:k+radius));
        yrefined(:,size(yraw, 3)-k+1) = yrefined(:,size(yraw,3)-k+1) ./ sum(lkernel(1:k+radius));
    end
    
    yrefined = reshape(yrefined, [size(yraw, 1) size(yraw, 2) size(yraw, 3)]);
end

end

