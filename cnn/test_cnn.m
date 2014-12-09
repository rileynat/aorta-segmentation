function [ ap_test, acc_test, test_pred ] = test_cnn( test_data, test_labels, weights, filterInfo )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    %[~, theta] = roll_params(0, weights);
    %[accuracy, ~] = gradient_cnn(theta, test_data, test_labels, filterInfo, 0);
    
    disp('Testing...');
    
    % FeedForward
    test_data = fit_HUscale(test_data);
    yhat_pred = FeedForwardCNN(test_data, weights, filterInfo);
    test_pred = yhat_pred;
    
    ap_test = zeros(size(test_data, 3), 1);
    acc_test = zeros(size(test_data,3), 1);
    for i = 1:size(test_data, 3)
       yhat = yhat_pred(:,:,i);
       yc = test_labels(:,:,i);
       [~, ~, ap] = compute_ap(yhat(:), yc(:));
       ap_test(i) =  ap;
       acc_test(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    end

end

