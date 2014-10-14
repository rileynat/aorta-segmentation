function [TESTSET, fname] = donothing(DEBUG, models, TRAINSET, VALSET, TESTSET, fnormalize, alpha)
%DONOTHING Summary of this function goes here
%   Detailed explanation goes here

fname = 'donothing';

if ~exist('DEBUG','var'),
    DEBUG = 1;
end

if ~exist('fnormalize','var'),
    fnormalize = @fit_HUscale;
end

if ~exist('alpha','var'),
    alpha = 0.0;
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
    
   TESTSET.ylist{i} = ys;
   if DEBUG,
       break;
   end
end

end

