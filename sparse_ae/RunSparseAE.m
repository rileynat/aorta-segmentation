function [features] = RunSparseAE(data,  saeInfo) 
    %% ==================================================
    %  |data| is the preprocessed, unlabled data
    %  This function will send |data| through the sparse autoencoder
    %  to learn the activations of the hidden units.
    %  
    %  The activation of the hidden units will be returned so that 
    %  it can be used as input to a CNN or other classification network.
    
    addpath('../common')
    addpath(genpath('utils/minFunc_2012'))
    
    theta = InitializeParameters(saeInfo.hiddenSize, saeInfo.inputSize);
    
    % Train the sparseae
    options.Method = 'lbfgs';
    options.maxIter = 20;
    options.display = 'on';
    
    [opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   saeInfo.inputSize, saeInfo.hiddenSize, ...
                                   saeInfo.lambda, saeInfo.sparsity, ...
                                   saeInfo.beta, data), ...
                              theta, options);
    
    % Checkout the weights --> do they look useful?
    %W1 = reshape(opttheta(1:saeInfo.hiddenSize * saeInfo.inputSize), saeInfo.hiddenSize, saeInfo.inputSize);
    %display_network(W1');
    
    features = feedForwardAutoencoder(opttheta, saeInfo, data);
        
    save('features');    

end
