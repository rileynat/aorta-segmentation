function [ numGrad ] = gradientChecking( costFunction, theta, grad )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    epsilon = 10^-4;
    n = size(theta,1);
    
    numGrad = zeros(size(theta));
    
    for i = 1:n
        thetaPlus = theta;
        thetaPlus(i) = theta(i) + epsilon;
        thetaMinus = theta;
        thetaMinus(i) = theta(i) - epsilon;
        costPlus = costFunction(thetaPlus);
        costMinus = costFunction(thetaMinus);
        numGrad(i) = (costPlus - costMinus) ./ (2 * epsilon);
        fprintf(2, '%d\n', numGrad(i) - grad(i));
    end
    
    diff = norm(numGrad-grad)/norm(numGrad+grad);
    disp(diff);

end

