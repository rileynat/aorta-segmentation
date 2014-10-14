function [cost, grad] = cnn_grad(x, y, weights, params)


% -- feed-forward inference
h = cnn_infer(x, weights, params);
yhat = cnn_recon(h, weights, params);


% -- compute cost
cost = cross_entropy(y, yhat);


% -- backprop
grad = replicate_struct(weights, 0);
dobj = (yhat - y);

% objective -> h
grad.visbias = grad.visbias + sum(dobj, 4);
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        grad.hidvis(:,:,c,b) = grad.hidvis(:,:,c,b) + convn(dobj(:,:,b,:), h(end:-1:1,end:-1:1,c,end:-1:1), 'valid');
    end
end

% dh
dh = zeros(size(h));
for b = 1:size(weights.hidvis, 4),
    for c = 1:size(weights.hidvis, 3),
        dh(:,:,c,:) = convn(dobj(:,:,b,:), weights.hidvis(end:-1:1,end:-1:1,c,b), 'valid');
    end
end
switch params.nonlinearity,
    case 'relu',
        dobj = dh.*(h > 0);
    case 'sigmoid',
        dobj = dh.*h.*(1-h);
end

% h -> input
grad.hidbias = grad.hidbias + permute(sum(sum(sum(dobj, 1), 2), 4), [3 1 2]);
for b = 1:size(weights.vishid, 4),
    for c = 1:size(weights.vishid, 3),
        grad.vishid(:,:,c,b) = grad.vishid(:,:,c,b) + convn(x(:,:,c,:), dobj(end:-1:1,end:-1:1,b,end:-1:1), 'valid');
    end
end


return;