function weights = cnn_train_lbfgs(xtrain, ytrain, params, xval, yval)


% -- initialization
weights = struct;
weights.vishid = 0.01*randn(params.ws, params.ws, params.numch, params.numhid);
weights.hidvis = 0.01*randn(params.ws, params.ws, params.numhid, params.numout);
weights.hidbias = zeros(params.numhid, 1);
weights.visbias = zeros(params.rs, params.cs, params.numout);


addpath(genpath('utils/minFunc_2012/'));
theta = cnn_roll(weights);


% lbfgs
options.method = 'lbfgs';
options.maxiter = params.maxiter;


opttheta = minFunc(@(p) cnn_grad_roll(p, xtrain, ytrain, params, xval, yval), theta, options);
weights = cnn_unroll(opttheta, params);


% -- filename to save
fname_mat = sprintf('models/%s.mat', params.fname);
save(fname_mat, 'weights', 'params');


return;