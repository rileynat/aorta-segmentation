% =====================================
% CNN feedforward inference
% =====================================

function h = cnn_infer(x, weights, params)


batchsize = size(x, 4);

vishidlr = zeros(params.ws, params.ws, params.numhid, params.numch);
for c = 1:params.numch,
    vishidlr(:,:,:,c) = reshape(weights.vishid(end:-1:1, end:-1:1, c, :),[params.ws,params.ws,params.numhid]);
end

hbiasmat = repmat(permute(weights.hidbias,[2 3 1]),[size(x,1)-params.ws+1, size(x,2)-params.ws+1, batchsize]);
hbiasmat = reshape(hbiasmat, [size(x,1)-params.ws+1, size(x,2)-params.ws+1, params.numhid, batchsize]);


h = hbiasmat;
if 0,
    % slow
    for c = 1:params.numch,
        for d = 1:params.numhid,
            for n = 1:batchsize,
                h(:,:,d,n) = h(:,:,d,n) + conv2(x(:,:,c,n), vishidlr(:,:,d,c), 'valid');
            end
        end
    end
else
    % fast
    for c = 1:params.numch,
        for d = 1:params.numhid,
            h(:,:,d,:) = h(:,:,d,:) + convn(x(:,:,c,:), vishidlr(:,:,d,c), 'valid');
        end
    end
end

switch params.nonlinearity,
    case 'relu',
        h = max(0, h);
    case 'sigmoid',
        h = sigmoid(h);
end


return;