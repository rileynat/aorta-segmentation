function [ TESTSET, fname ] = donothing_except_output_avg(DEBUG, models, TRAINSET, VALSET, TESTSET, fnormalize, alpha)
%DONOTHING_EXCEPT_OUTPUT_AVG Summary of this function goes here
%   Detailed explanation goes here

fname = 'donothing_except_output_avg';

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
   
   ysmooth = zeros(size(ys));
   for j = 1:seqL,
       ratios = 0.0;
       for k = max(1, j-2):min(seqL, j+2)
           cur_r = exp(-abs(k-j)*1.0);
           ratios = ratios + cur_r;
           ysmooth(:,:,j) = ysmooth(:,:,j) + cur_r.*ys(:,:,k);
       end
       ysmooth(:,:,j) = ysmooth(:,:,j) ./ ratios;
   end
   
    
   TESTSET.ylist{i} = ysmooth;
   if DEBUG,
       break;
   end
end

end



