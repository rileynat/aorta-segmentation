function [ y ] = sigmoid( x )
%sigmoid computes the element-wise sigmoid function of x

    y = 1 ./ (1 + exp(-x));

end

