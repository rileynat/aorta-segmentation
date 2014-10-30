function [ VALSET, TESTSET, fname, filterspec ] = naive_smoothing(DEBUG, TRAINSET, VALSET, TESTSET, models, fnormalize, filterparams)
%DONOTHING_EXCEPT_OUTPUT_AVG Summary of this function goes here
%   Detailed explanation goes here

fname = 'naive_smoothing';

if ~exist('DEBUG','var'),
    DEBUG = 1;
end

if ~exist('fnormalize','var'),
    fnormalize = @fit_HUscale;
end

if ~exist('filterparams','var'),
    filterparams = struct('filtername', 'laplace', 'lambda', 0.1, 'winsize', 9);
end

if (strcmp(filterparams.filtername, 'laplace') == 1),
    func_filter = @laplacian_filter;
else (strcmp(filterparams.filtername, 'gauss') == 1),
    func_filter = @gaussian_filter;
end

filterspec = sprintf('%s_lambda%g_winsize%d',filterparams.filtername, filterparams.lambda, filterparams.winsize);

for i = 1:VALSET.num,
    xs = VALSET.xlist{i};
    refz = VALSET.refzlist{i};
    seqL = size(xs, 3);
    ys = [];
    
    for j = 1:seqL
        xc = xs(:,:,j);
        xc = fnormalize(xc);
        idx = refz(j);
        h = cnn_infer(xc, models{idx}.weights, models{idx}.params);
      yhat = cnn_recon(h, models{idx}.weights, models{idx}.params);
      ys = cat(3, ys, yhat);
   end
   
   VALSET.ylist{i} = func_filter(ys, filterparams);
   if DEBUG,
       break;
   end
end

for i = 1:TESTSET.num,
   xs = TESTSET.xlist{i};
   refz = TESTSET.refzlist{i};
   seqL = size(xs, 3);
   ys = [];
   
   for j = 1:seqL
      xc = xs(:,:,j);
      xc = fnormalize(xc);
      idx = refz(j);
      h = cnn_infer(xc, models{idx}.weights, models{idx}.params);
      yhat = cnn_recon(h, models{idx}.weights, models{idx}.params);
      ys = cat(3, ys, yhat);
   end
   
   TESTSET.ylist{i} = func_filter(ys, filterparams);
   if DEBUG,
       break;
   end
end

end



