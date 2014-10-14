function y = cnn_recon(h, weights, params)

batchsize = size(h, 4);

hidvis = weights.hidvis;

vbiasmat = repmat(weights.visbias, [1 1 1 batchsize]);

y = vbiasmat;
if 0,
    % slow
    for n = 1:batchsize,
        for b = 1:params.numhid,
            for c = 1:params.numout,
                y(:,:,c,n) = y(:,:,c,n) + conv2(h(:,:,b,n), hidvis(:,:,b,c), 'full');
            end
        end
    end
else
    % fast
    for b = 1:params.numhid,
        for c = 1:params.numout,
            y(:,:,c,:) = y(:,:,c,:) + convn(h(:,:,b,:), hidvis(:,:,b,c), 'full');
        end
    end
end

y = sigmoid(y);

return;