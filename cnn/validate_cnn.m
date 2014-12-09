function [ ap_val, acc_val, val_pred ] = validate_cnn( val_data, val_labels, weights, filterInfo )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %[~, theta] = roll_params(0, weights);
    %[accuracy, ~] = gradient_cnn(theta, val_data, val_labels, filterInfo, 0);
    
    disp('Validating...');
    
    % FeedForward
    val_data = fit_HUscale(val_data);
    yhat_pred = FeedForwardCNN(val_data, weights, filterInfo);
    val_pred = yhat_pred;
    
    ap_val = zeros(size(val_data, 3), 1);
    acc_val = zeros(size(val_data,3), 1);
    for i = 1:size(val_data, 3)
       yhat = yhat_pred(:,:,i);
       yc = val_labels(:,:,i);
       [~, ~, ap] = compute_ap(yhat(:), yc(:));
       ap_val(i) =  ap;
       acc_val(i) = 1-mean(yc(:) ~= (yhat(:) > 0.5));
    end
    
    
end

