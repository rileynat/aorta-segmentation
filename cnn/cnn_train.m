function [weights, history] = cnn_train(xtrain, ytrain, params, xval, yval)


% -- initialization
weights = struct;
weights.vishid = 0.01*randn(params.ws, params.ws, params.numch, params.numhid);
weights.hidvis = 0.01*randn(params.ws, params.ws, params.numhid, params.numout);
weights.hidbias = zeros(params.numhid, 1);
weights.visbias = zeros(params.rs, params.cs, params.numout);


% -- structs for gradients
grad = replicate_struct(weights, 0);
neg = replicate_struct(weights, 0);


% -- filename to save
fname_mat = sprintf('models/%s.mat', params.fname);


% -- monitoring variables, etc
batchsize = params.batchsize;
maxiter = params.maxiter;
history.cost = zeros(maxiter, 1);


% -- start training
N = size(xtrain, 4);
numbatch = min(floor(N/batchsize), 100);

for t = 1:maxiter,
    if t > params.momentum_change,
        momentum = params.momentum_final;
    else
        momentum = params.momentum_init;
    end
    
    epsilon = params.eps/(1+params.eps_decay*t);
    
    cost_epoch = zeros(numbatch, 1);
    randidx = randperm(N);
    
    
    ts_epoch = tic;
    for b = 1:numbatch,
        batchidx = randidx((b-1)*batchsize+1:b*batchsize);
        xb = xtrain(:, :, :, batchidx);
        yb = ytrain(:, :, :, batchidx);
        
        ts = tic;
        [cost, pos] = cnn_grad(xb, yb, weights, params);
        cost_epoch(b) = cost/batchsize;
        te = toc(ts);
        
        if params.verbose,
            fprintf('epoch [%d/%d] batch [%d/%d] : cost = %g (time = %g)\n', t, maxiter, b, numbatch, cost_epoch(b), te);
        end
        
        pos = replicate_struct(pos, -1/batchsize);
        
        [weights, grad] = update_params(weights, grad, pos, neg, momentum, epsilon, 0);
    end
    te_epoch = toc(ts_epoch);
    history.cost(t) = mean(cost_epoch);
    
    fprintf('epoch [%d/%d] : cost = %g (time per epoch = %g)\n', t, maxiter, history.cost(t), te_epoch);
    
    if exist('xval', 'var'),
        [~, ~, ap] = cnn_evaluate(xval, yval, weights, params);
        history.val_ap(t) = ap;
        fprintf('epoch [%d/%d] : val AP = %g\n', t, maxiter, ap);
    end
    
    [weights, grad] = save_params(fname_mat, weights, grad, params, t, history);
end


return;